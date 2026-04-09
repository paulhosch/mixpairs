from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from . import renderers

_BIVAR_REGISTRY: dict[str, Callable[..., Any]] = {}
_UNIVAR_REGISTRY: dict[str, Callable[..., Any]] = {}

_BIVAR_REQUIRED = ["ax", "data", "x_col", "y_col", "hue_col", "palette", "orient"]
_UNIVAR_REQUIRED = ["ax", "data", "col", "hue_col", "palette"]


def _validate_signature(func: Callable[..., Any], kind: str) -> None:
    sig = inspect.signature(func)
    params = sig.parameters
    required = _BIVAR_REQUIRED if kind == "bivar" else _UNIVAR_REQUIRED
    for name in required:
        if name not in params:
            raise TypeError(f"Renderer '{func.__name__}' missing required parameter '{name}'.")
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if not has_kwargs:
        raise TypeError(f"Renderer '{func.__name__}' must accept **kwargs.")


def register_renderer(name: str, func: Callable[..., Any], kind: str = "bivar") -> None:
    if kind not in {"bivar", "univar"}:
        raise ValueError("kind must be 'bivar' or 'univar'.")
    _validate_signature(func, kind)
    registry = _BIVAR_REGISTRY if kind == "bivar" else _UNIVAR_REGISTRY
    registry[name] = func


def resolve_renderer(key: str | Callable[..., Any] | None, kind: str) -> Callable[..., Any]:
    if kind not in {"bivar", "univar"}:
        raise ValueError("kind must be 'bivar' or 'univar'.")
    if key in {None, "blank"}:
        return renderers._blank_renderer
    if callable(key):
        _validate_signature(key, kind)
        return key
    if not isinstance(key, str):
        raise TypeError("renderer key must be str, callable, None, or 'blank'.")
    registry = _BIVAR_REGISTRY if kind == "bivar" else _UNIVAR_REGISTRY
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise KeyError(f"Unknown renderer '{key}'. Available: {available}")
    return registry[key]


def _register_builtins() -> None:
    register_renderer("scatter", renderers.scatter, kind="bivar")
    register_renderer("scatter_reg", renderers.scatter_reg, kind="bivar")
    register_renderer("kde2d", renderers.kde2d, kind="bivar")
    register_renderer("hexbin", renderers.hexbin, kind="bivar")
    register_renderer("corr_text", renderers.corr_text, kind="bivar")
    register_renderer("box", renderers.box, kind="bivar")
    register_renderer("violin", renderers.violin, kind="bivar")
    register_renderer("strip", renderers.strip, kind="bivar")
    register_renderer("facet_hist", renderers.facet_hist, kind="bivar")
    register_renderer("facet_kde", renderers.facet_kde, kind="bivar")
    register_renderer("count_heatmap", renderers.count_heatmap, kind="bivar")
    register_renderer("stacked_bar", renderers.stacked_bar, kind="bivar")

    register_renderer("hist_diag", renderers.hist_diag, kind="univar")
    register_renderer("kde_diag", renderers.kde_diag, kind="univar")
    register_renderer("bar_diag", renderers.bar_diag, kind="univar")


_register_builtins()
