import torch
from owl_wms.models.world import CondHead


class CachedCondHead(CondHead):
    def __init__(self, cond_head_inner, config):
        super().__init__()

        # determine input shapes
        num_schedule_steps = len(config.scheduler_sigmas) - 1
        out_shape = config.d_model

        self.inner = cond_head_inner
        self.register_buffer("cache", torch.empty(num_schedule_steps, *out_shape))
        self.register_buffer("filled", torch.zeros(num_schedule_steps, dtype=torch.bool))

    def forward(self, x):
        key = x.view(-1).sum()
        mask = ~self.filled[key]
        idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)

        if idx.numel() > 0:  # <- small Python branch
            y_new = super().forward(x[idx])
            k_new = key[idx]
            self.cache.index_copy_(0, k_new, y_new)
            self.filled.index_fill_(0, k_new, True)

        return self.cache[key]


def patch_CondHead_CachedCondHead(model):
    for name, mod in list(model.named_modules()):
        if isinstance(mod, CondHead):
            model.set_submodule(name, CachedCondHead(mod, model.config))


def apply_inference_patches(model):
    patch_CondHead_CachedCondHead(model)
