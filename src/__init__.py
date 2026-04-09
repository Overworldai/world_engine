from .world_engine import WorldEngine, CtrlInput
from .model.world_model import WorldModel
from .quantize import QUANTS


def __getattr__(name):
    # Lazy import: MLXWorldEngine depends on mlx which is macOS-only.
    # Deferred so that `from world_engine import WorldEngine` works on all platforms.
    if name == "MLXWorldEngine":
        from .mlx_metal.engine import MLXWorldEngine
        return MLXWorldEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["WorldEngine", "MLXWorldEngine", "CtrlInput", "WorldModel", "QUANTS"]
