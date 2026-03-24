"""
OpenVINO-based WorldEngine for Intel GPU inference.
All passes (denoise + cache) on GPU. Pre-allocated I/O buffers for minimal overhead.
"""
from typing import Optional, Set, Tuple, Dict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import openvino as ov
    HAS_OPENVINO = True
except ImportError:
    HAS_OPENVINO = False

import torch
from torch import Tensor

from .model.world_model import WorldModel, PromptEncoder
from .ae import get_ae
from .portable_model import PortableWorldModel
from .stateless_kv import StatelessKVManager


@dataclass
class CtrlInput:
    button: Set[int] = field(default_factory=set)
    mouse: Tuple[float, float] = (0.0, 0.0)
    scroll_wheel: int = 0


class OpenVINOWorldEngine:
    def __init__(
        self,
        model_uri: str,
        mode: str = "portable",
        ir_dir: Optional[str] = None,
        device: str = "GPU",
        dtype=torch.float32,
        load_weights: bool = True,
    ):
        self.mode = mode
        self.ov_device = device
        self.dtype = dtype
        self.model_cfg = WorldModel.load_config(model_uri)

        self.vae = get_ae(
            self.model_cfg.ae_uri,
            is_taehv_ae=getattr(self.model_cfg, "taehv_ae", False),
            auto_aspect_ratio=getattr(self.model_cfg, "auto_aspect_ratio", True),
            device="cpu", dtype=dtype,
        )

        self.prompt_encoder = None
        if self.model_cfg.prompt_conditioning is not None:
            pe_uri = getattr(self.model_cfg, "prompt_encoder_uri", "google/umt5-xl")
            self.prompt_encoder = PromptEncoder(pe_uri, dtype=dtype).eval()
            self.prompt_encoder.encode = lambda inputs: self.prompt_encoder.encoder(**inputs).last_hidden_state

        self.scheduler_sigmas = torch.tensor(self.model_cfg.scheduler_sigmas, device="cpu", dtype=dtype)

        pH, pW = tuple(self.model_cfg.patch)
        self.frm_shape = (1, 1, self.model_cfg.channels, self.model_cfg.height * pH, self.model_cfg.width * pW)

        inference_fps = getattr(self.model_cfg, "inference_fps", self.model_cfg.base_fps)
        latent_fps = inference_fps / getattr(self.model_cfg, "temporal_compression", 1)
        self.ts_mult = int(self.model_cfg.base_fps) // latent_fps

        kv_dtype = torch.float16 if mode == "openvino" else dtype
        self.kv_mgr = StatelessKVManager(self.model_cfg, batch_size=1, dtype=kv_dtype, device=torch.device("cpu"))
        self.frame_ts = torch.tensor([[0]], dtype=torch.long)
        self._prompt_emb = None
        self._prompt_pad_mask = None

        if mode == "portable":
            self._init_portable(model_uri, load_weights)
        elif mode == "openvino":
            self._init_openvino(ir_dir)

    def _init_portable(self, model_uri: str, load_weights: bool):
        if load_weights:
            original = WorldModel.from_pretrained(model_uri, cfg=self.model_cfg, device="cpu", dtype=self.dtype)
        else:
            original = WorldModel(self.model_cfg).to("cpu").to(self.dtype)
        self.model = PortableWorldModel.from_original(original.eval()).to(self.dtype).eval()
        del original

    def _init_openvino(self, ir_dir: Optional[str]):
        if not HAS_OPENVINO:
            raise ImportError("openvino not installed")
        ir_path = Path(ir_dir)
        core = ov.Core()

        available = core.available_devices
        print(f"OpenVINO available devices: {available}")
        is_compound = any(self.ov_device.startswith(p) for p in ("AUTO", "HETERO", "MULTI"))
        if not is_compound and self.ov_device not in available:
            print(f"Warning: {self.ov_device} not available, falling back to CPU")
            self.ov_device = "CPU"

        gpu_config = {
            "CACHE_DIR": str(ir_path / ".ov_cache"),
            "PERFORMANCE_HINT": "LATENCY",
            "INFERENCE_PRECISION_HINT": "f16",
        }

        # Find best model
        model_path = None
        for candidate in [
            *sorted(ir_path.glob("transformer_int4*.xml"), reverse=True),
            ir_path / "transformer.xml",
            *sorted(ir_path.glob("transformer_frozen_int4*.xml"), reverse=True),
            ir_path / "transformer_frozen.xml",
        ]:
            if candidate.exists():
                model_path = candidate
                break

        self.ov_model = None
        self._is_frozen_only = False

        if model_path is not None:
            self._is_frozen_only = "frozen" in model_path.name
            print(f"Loading {model_path.name} on {self.ov_device}...")
            self.ov_model = core.compile_model(str(model_path), self.ov_device, gpu_config)
            print(f"  Inputs: {len(self.ov_model.inputs)}, Outputs: {len(self.ov_model.outputs)}")
            print(f"  Device: {self.ov_model.get_property('EXECUTION_DEVICES')}")
            self._setup_io_buffers()

        # VAE
        self.ov_vae_dec = None
        self.ov_vae_enc = None
        if (ir_path / "vae_decoder.xml").exists():
            self.ov_vae_dec = core.compile_model(str(ir_path / "vae_decoder.xml"), self.ov_device)
        if (ir_path / "vae_encoder.xml").exists():
            self.ov_vae_enc = core.compile_model(str(ir_path / "vae_encoder.xml"), self.ov_device)

    def _setup_io_buffers(self):
        """Pre-allocate all input/output numpy buffers and bind to infer request.
        This eliminates ov.Tensor creation overhead per call (~0.5s saved per infer)."""
        req = self.ov_model.create_infer_request()
        self._req = req
        n_layers = self.model_cfg.n_layers
        off = 0 if self._is_frozen_only else 1

        # Pre-allocate input buffers as numpy arrays
        C = self.model_cfg.channels
        pH, pW = tuple(self.model_cfg.patch)
        H, W = self.model_cfg.height * pH, self.model_cfg.width * pW

        self._buf_x = np.zeros((1, 1, C, H, W), dtype=np.float32)
        self._buf_sigma = np.zeros((1, 1), dtype=np.float32)
        self._buf_frame_ts = np.zeros((1, 1), dtype=np.int64)
        self._buf_frame_idx = np.zeros((1, 1), dtype=np.int64)
        self._buf_mouse = np.zeros((1, 1, 2), dtype=np.float32)
        self._buf_button = np.zeros((1, 1, self.model_cfg.n_buttons), dtype=np.float32)
        self._buf_scroll = np.zeros((1, 1, 1), dtype=np.float32)

        prompt_dim = getattr(self.model_cfg, 'prompt_embedding_dim', 1)
        self._buf_prompt = np.zeros((1, 512, prompt_dim), dtype=np.float32)
        self._buf_prompt_mask = np.ones((1, 512), dtype=bool)

        if not self._is_frozen_only:
            self._buf_is_frozen = np.array([True], dtype=bool)

        # Bind all input tensors ONCE — these ov.Tensor objects wrap the numpy buffers
        # and persist for the lifetime of the engine
        req.set_input_tensor(0, ov.Tensor(self._buf_x, shared_memory=True))
        req.set_input_tensor(1, ov.Tensor(self._buf_sigma, shared_memory=True))
        req.set_input_tensor(2, ov.Tensor(self._buf_frame_ts, shared_memory=True))
        req.set_input_tensor(3, ov.Tensor(self._buf_frame_idx, shared_memory=True))
        if not self._is_frozen_only:
            req.set_input_tensor(4, ov.Tensor(self._buf_is_frozen, shared_memory=True))
        req.set_input_tensor(4 + off, ov.Tensor(self._buf_mouse, shared_memory=True))
        req.set_input_tensor(5 + off, ov.Tensor(self._buf_button, shared_memory=True))
        req.set_input_tensor(6 + off, ov.Tensor(self._buf_scroll, shared_memory=True))
        req.set_input_tensor(7 + off, ov.Tensor(self._buf_prompt, shared_memory=True))
        req.set_input_tensor(8 + off, ov.Tensor(self._buf_prompt_mask, shared_memory=True))

        # KV buffers — bind to the torch tensor's underlying numpy memory
        base_kv = 9 + off
        kv_bufs, written_bufs = self.kv_mgr.get_state()
        self._kv_input_tensors = []
        self._written_input_tensors = []
        for i in range(n_layers):
            kv_ov = ov.Tensor(kv_bufs[i].numpy(), shared_memory=True)
            req.set_input_tensor(base_kv + i, kv_ov)
            self._kv_input_tensors.append(kv_ov)
        for i in range(n_layers):
            w_ov = ov.Tensor(written_bufs[i].numpy(), shared_memory=True)
            req.set_input_tensor(base_kv + n_layers + i, w_ov)
            self._written_input_tensors.append(w_ov)

    def _rebind_kv_inputs(self):
        """Rebind KV input tensors after KV state changes (cache pass output)."""
        n_layers = self.model_cfg.n_layers
        off = 0 if self._is_frozen_only else 1
        base_kv = 9 + off
        kv_bufs, written_bufs = self.kv_mgr.get_state()
        for i in range(n_layers):
            kv_ov = ov.Tensor(kv_bufs[i].numpy(), shared_memory=True)
            self._req.set_input_tensor(base_kv + i, kv_ov)
            self._kv_input_tensors[i] = kv_ov
        for i in range(n_layers):
            w_ov = ov.Tensor(written_bufs[i].numpy(), shared_memory=True)
            self._req.set_input_tensor(base_kv + n_layers + i, w_ov)
            self._written_input_tensors[i] = w_ov

    def reset(self):
        self.kv_mgr.reset()
        self.frame_ts.zero_()
        self.vae.reset()
        if hasattr(self, '_req'):
            self._rebind_kv_inputs()

    @torch.inference_mode()
    def get_state(self):
        kv_bufs, written_bufs = self.kv_mgr.get_state()
        return {
            "kv_bufs": [kv.detach().clone() for kv in kv_bufs],
            "written_bufs": [w.detach().clone() for w in written_bufs],
            "frame_ts": self.frame_ts.detach().clone(),
        }

    @torch.inference_mode()
    def load_state(self, state):
        self.kv_mgr.set_state(state["kv_bufs"], state["written_bufs"])
        self.frame_ts.copy_(state["frame_ts"])
        if hasattr(self, '_req'):
            self._rebind_kv_inputs()

    def set_prompt(self, prompt: str):
        if self.prompt_encoder is None:
            raise RuntimeError("prompt_conditioning not configured")
        with torch.inference_mode():
            emb, mask = self.prompt_encoder([prompt])
            self._prompt_emb = emb
            self._prompt_pad_mask = mask
            if hasattr(self, '_buf_prompt'):
                np.copyto(self._buf_prompt, emb.numpy())
                np.copyto(self._buf_prompt_mask, mask.numpy())

    def _prep_ctrl(self, ctrl: Optional[CtrlInput] = None):
        ctrl = ctrl or CtrlInput()
        button = torch.zeros(1, 1, self.model_cfg.n_buttons, dtype=self.dtype)
        if ctrl.button:
            button[..., list(ctrl.button)] = 1.0
        mouse = torch.as_tensor(ctrl.mouse, dtype=self.dtype).reshape(1, 1, 2)
        scroll = torch.sign(torch.as_tensor(ctrl.scroll_wheel, dtype=self.dtype)).reshape(1, 1, 1)
        return mouse, button, scroll

    @torch.inference_mode()
    def gen_frame(self, ctrl: Optional[CtrlInput] = None, return_img: bool = True):
        if self.model_cfg.prompt_conditioning is not None and self._prompt_emb is None:
            self.set_prompt("An explorable world")

        x = torch.randn(self.frm_shape, dtype=self.dtype)
        mouse, button, scroll = self._prep_ctrl(ctrl)

        x0 = self._denoise_pass(x, mouse, button, scroll)
        self._cache_pass(x0, mouse, button, scroll)

        if return_img:
            return self._decode(x0.squeeze(1))
        return x0.squeeze(1)

    @torch.inference_mode()
    def append_frame(self, img: Tensor, ctrl: Optional[CtrlInput] = None):
        assert img.dtype == torch.uint8
        x0 = self._encode(img).unsqueeze(1)
        mouse, button, scroll = self._prep_ctrl(ctrl)
        self._cache_pass(x0, mouse, button, scroll)
        return img

    def _get_frame_timestamps(self):
        frame_idx = self.frame_ts.clone()
        frame_timestamp = self.frame_ts * int(self.ts_mult)
        return frame_idx, frame_timestamp

    def _denoise_pass(self, x, mouse, button, scroll):
        """Denoise loop on GPU. Pre-bound buffers — only write data in-place, no ov.Tensor creation."""
        frame_idx, frame_timestamp = self._get_frame_timestamps()

        if self.mode == "openvino" and self.ov_model is not None:
            # Write static inputs into pre-allocated buffers (just numpy copies, no ov.Tensor creation)
            self._buf_frame_ts[0, 0] = frame_timestamp.item()
            self._buf_frame_idx[0, 0] = frame_idx.item()
            np.copyto(self._buf_mouse, mouse.numpy())
            np.copyto(self._buf_button, button.numpy())
            np.copyto(self._buf_scroll, scroll.numpy())
            if not self._is_frozen_only:
                self._buf_is_frozen[0] = True

            # Denoise steps — only update x and sigma in-place, use async for lower latency
            sigmas = self.scheduler_sigmas
            dsigmas = sigmas.diff()
            for step_sig, step_dsig in zip(sigmas, dsigmas):
                np.copyto(self._buf_x, x.numpy())
                self._buf_sigma[0, 0] = step_sig.item()
                self._req.start_async()
                self._req.wait()
                v_pred_np = self._req.get_output_tensor(0).data
                x = x + step_dsig.item() * torch.from_numpy(v_pred_np)
        else:
            kv_bufs, written_bufs = self.kv_mgr.get_state()
            is_frozen = torch.tensor([True], dtype=torch.bool)
            for step_sig, step_dsig in zip(self.scheduler_sigmas, self.scheduler_sigmas.diff()):
                sigma = torch.full((1, 1), step_sig.item(), dtype=self.dtype)
                v_pred, _, _ = self.model(
                    x, sigma, frame_timestamp, kv_bufs, written_bufs, is_frozen,
                    frame_idx=frame_idx, prompt_emb=self._prompt_emb,
                    prompt_pad_mask=self._prompt_pad_mask,
                    mouse=mouse, button=button, scroll=scroll,
                )
                x = x + step_dsig * v_pred
        return x

    def _cache_pass(self, x0, mouse, button, scroll):
        """Cache pass on GPU. Updates KV state."""
        frame_idx, frame_timestamp = self._get_frame_timestamps()

        if self.mode == "openvino" and self.ov_model is not None and not self._is_frozen_only:
            # Write inputs into pre-allocated buffers
            np.copyto(self._buf_x, x0.numpy())
            self._buf_sigma[0, 0] = 0.0
            self._buf_frame_ts[0, 0] = frame_timestamp.item()
            self._buf_frame_idx[0, 0] = frame_idx.item()
            np.copyto(self._buf_mouse, mouse.numpy())
            np.copyto(self._buf_button, button.numpy())
            np.copyto(self._buf_scroll, scroll.numpy())
            self._buf_is_frozen[0] = False

            self._req.start_async()
            self._req.wait()

            # Copy KV outputs into KV manager buffers, then rebind
            n_layers = self.model_cfg.n_layers
            kv_bufs, written_bufs = self.kv_mgr.get_state()
            for i in range(n_layers):
                np.copyto(kv_bufs[i].numpy(), self._req.get_output_tensor(1 + i).data)
                np.copyto(written_bufs[i].numpy(), self._req.get_output_tensor(1 + n_layers + i).data)
            # KV data changed in the torch tensors' memory — rebind ov.Tensors
            # Actually since we copied INTO the same numpy buffers that the ov.Tensors point to,
            # we don't need to rebind. The shared_memory ov.Tensors still point to the same memory.
            # BUT: the output tensors wrote to OV's internal buffers, and we copyto'd into the
            # input buffers. So the input ov.Tensors are already updated. No rebind needed.
        else:
            kv_bufs, written_bufs = self.kv_mgr.get_state()
            is_frozen = torch.tensor([False], dtype=torch.bool)
            sigma = torch.zeros(1, 1, dtype=self.dtype)
            _, kv_bufs_out, written_bufs_out = self.model(
                x0, sigma, frame_timestamp, kv_bufs, written_bufs, is_frozen,
                frame_idx=frame_idx, prompt_emb=self._prompt_emb,
                prompt_pad_mask=self._prompt_pad_mask,
                mouse=mouse, button=button, scroll=scroll,
            )
            self.kv_mgr.set_state(kv_bufs_out, written_bufs_out)

        self.frame_ts.add_(1)

    def _decode(self, latent: Tensor) -> Tensor:
        if self.ov_vae_dec is not None:
            result = self.ov_vae_dec(latent.numpy())
            decoded = torch.from_numpy(result[0])
            decoded = (decoded / 2 + 0.5).clamp(0, 1)
            decoded = (decoded * 255).round().to(torch.uint8)
            return decoded.squeeze(0).permute(1, 2, 0)[..., :3]
        else:
            with torch.inference_mode():
                decoded = self.vae.ae_model.decoder(latent.to(self.dtype))
                decoded = (decoded / 2 + 0.5).clamp(0, 1)
                decoded = (decoded * 255).round().to(torch.uint8)
                return decoded.squeeze(0).permute(1, 2, 0)[..., :3]

    def _encode(self, img: Tensor) -> Tensor:
        if self.ov_vae_enc is not None:
            img_f = img.unsqueeze(0).to(self.dtype).permute(0, 3, 1, 2).div(255).mul(2).sub(1)
            in_ch = self.vae.ae_model.encoder.conv_in.proj.in_channels
            if img_f.shape[1] < in_ch:
                pad = torch.zeros(img_f.shape[0], in_ch - img_f.shape[1],
                                  img_f.shape[2], img_f.shape[3], dtype=self.dtype)
                img_f = torch.cat([img_f, pad], dim=1)
            result = self.ov_vae_enc(img_f.numpy())
            return torch.from_numpy(result[0])
        else:
            return self.vae.encode(img)
