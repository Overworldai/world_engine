import warnings
import torch


class NoCastModule(torch.nn.Module):
    def _apply(self, fn):
        def keep_dtype(t):
            old_dtype = t.dtype
            out = fn(t)
            if out.dtype is not old_dtype:
                warnings.warn(
                    f"{self.__class__.__name__}: requested dtype cast ignored; "
                    f"keeping {old_dtype}.",
                    stacklevel=3,
                )
                out = out.to(dtype=old_dtype)
            return out

        return super()._apply(keep_dtype)

    def to(self, *args, **kwargs):
        warn_cast = False

        # m.to(ref_tensor): use ref's device, ignore its dtype
        if args and isinstance(args[0], torch.Tensor):
            ref, *rest = args
            args = (ref.device, *rest)
            base = next(self.parameters(), None) or next(self.buffers(), None)
            if base is not None and ref.dtype is not base.dtype:
                warn_cast = True

        # keyword dtype
        if kwargs.pop("dtype", None) is not None:
            warn_cast = True

        # positional dtype
        args = tuple(a for a in args if not isinstance(a, torch.dtype))

        if warn_cast:
            warnings.warn(
                f"{self.__class__.__name__}.to: requested dtype cast ignored; "
                "keeping existing dtypes.",
                stacklevel=2,
            )

        return super().to(*args, **kwargs)
