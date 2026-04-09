from ._registry import register_renderer
from .config import DType, Orient, PairGridConfig, PairType, SectionConfig, Triangle
from .core import ggpairs

__all__ = [
    "ggpairs",
    "register_renderer",
    "DType",
    "PairType",
    "Triangle",
    "Orient",
    "SectionConfig",
    "PairGridConfig",
]
