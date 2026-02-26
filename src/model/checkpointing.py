import torch
from contextlib import nullcontext, ExitStack, contextmanager
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.autograd.graph import saved_tensors_hooks


def checkpoint_gpu(fn, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch_checkpoint(fn, *args, **kwargs)


class _SaveFirstActivation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Make sure "x" becomes the first tensor saved-for-backward in this region
        ctx.save_for_backward(x)
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad


@torch._disable_dynamo
def checkpoint_cpu(fn, *args, **kwargs):
    # Use non-reentrant to play nice with DDP.
    kwargs.setdefault("use_reentrant", False)
    preserve = kwargs.pop("preserve_rng_state", True)

    # Extract checkpoint-only kwargs (so they don't get passed to fn)
    _ = kwargs.pop("use_reentrant", None)
    user_context_fn = kwargs.pop("context_fn", None)
    determinism_check = kwargs.pop("determinism_check", None)
    debug = kwargs.pop("debug", None)

    if not args or not torch.is_tensor(args[0]):
        raise TypeError("CPU checkpoint expects first positional arg to be a Tensor (hidden_states/x).")

    fn_kwargs = dict(kwargs)

    def wrapped(x, *rest):
        # Force-save first activation, then call user fn
        x = _SaveFirstActivation.apply(x)
        if fn_kwargs:
            return fn(x, *rest, **fn_kwargs)
        return fn(x, *rest)

    def offload_only_first_saved_tensor_context():
        seen = {"done": False}

        def pack(t: torch.Tensor):
            if (not seen["done"]) and torch.is_tensor(t):
                seen["done"] = True
                # OFFLOAD to regular CPU memory (NOT pinned)
                cpu = t.detach().to("cpu")
                return ("cpu", cpu, t.device)
            # Keep everything else as-is (no offload)
            return ("gpu", t, None)

        def unpack(packed):
            tag, val, dev = packed
            if tag == "cpu":
                # Move back to original device (non_blocking won't help without pinning)
                return val.to(dev, non_blocking=False)
            return val

        return saved_tensors_hooks(pack, unpack)

    @contextmanager
    def combined(*cms):
        with ExitStack() as stack:
            for cm in cms:
                stack.enter_context(cm)
            yield

    def context_fn():
        base_fwd = offload_only_first_saved_tensor_context()
        base_recompute = nullcontext()

        if user_context_fn is None:
            return base_fwd, base_recompute

        u_fwd, u_recompute = user_context_fn()
        return combined(base_fwd, u_fwd), combined(base_recompute, u_recompute)

    ckpt_kwargs = {}
    if determinism_check is not None:
        ckpt_kwargs["determinism_check"] = determinism_check
    if debug is not None:
        ckpt_kwargs["debug"] = debug

    return torch_checkpoint(
        wrapped,
        *args,
        preserve_rng_state=preserve,
        use_reentrant=False,
        context_fn=context_fn,
        **ckpt_kwargs,
    )


def maybe_ckpt(mode, fn, *args, **kwargs):
    if mode in (None, False, "none"):
        return fn(*args, **kwargs)
    if mode in (True, "gpu"):
        return checkpoint_gpu(fn, *args, **kwargs)
    if mode == "cpu":
        return checkpoint_cpu(fn, *args, **kwargs)
    raise ValueError(f"mode must be one of: none|gpu|cpu (got {mode!r})")