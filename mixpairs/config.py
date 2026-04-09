from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable


class DType(Enum):
    CONTINUOUS = "continuous"
    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"


class PairType(Enum):
    CONTINUOUS = "continuous"
    COMBO = "combo"
    DISCRETE = "discrete"
    DIAG_CONTINUOUS = "diag_continuous"
    DIAG_CATEGORICAL = "diag_categorical"


class Triangle(Enum):
    UPPER = "upper"
    LOWER = "lower"
    DIAG = "diag"


class Orient(Enum):
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


RendererKey = str | Callable[..., Any] | None


@dataclass(frozen=True)
class SectionConfig:
    continuous: RendererKey = None
    combo: RendererKey = None
    discrete: RendererKey = None


@dataclass(frozen=True)
class PairGridConfig:
    upper: SectionConfig = field(
        default_factory=lambda: SectionConfig(
            continuous="corr_text",
            combo="box",
            discrete="count_heatmap",
        )
    )
    lower: SectionConfig = field(
        default_factory=lambda: SectionConfig(
            continuous="scatter",
            combo="facet_hist",
            discrete="stacked_bar",
        )
    )
    diag: SectionConfig = field(
        default_factory=lambda: SectionConfig(
            continuous="kde_diag",
            discrete="bar_diag",
        )
    )
    hue: str | None = None
    palette: str | list | dict | None = None
    cardinality_threshold: int = 15
    height: float = 2.0
    aspect: float = 1.0
    corner: bool = False
    share_axes: bool = True
    title: str | None = None
    column_labels: dict[str, str] | None = None
    dtypes: dict[str, DType] | None = None
    sort_by_type: bool = False
    subsample: int | None = None
    dropna_hue: bool = True
    legend_position: str = "right"
    font_scale: float = 1.0


def get_preset(name: str) -> PairGridConfig:
    key = name.lower()
    if key == "default":
        return PairGridConfig()
    if key == "minimal":
        return PairGridConfig(
            upper=SectionConfig(continuous="corr_text", combo="blank", discrete="blank"),
            lower=SectionConfig(continuous="scatter", combo="box", discrete="count_heatmap"),
            diag=SectionConfig(continuous="hist_diag", discrete="bar_diag"),
            legend_position="none",
        )
    if key == "kde":
        return PairGridConfig(
            upper=SectionConfig(continuous="kde2d", combo="facet_kde", discrete="count_heatmap"),
            lower=SectionConfig(continuous="kde2d", combo="facet_kde", discrete="stacked_bar"),
            diag=SectionConfig(continuous="kde_diag", discrete="bar_diag"),
        )
    if key == "regression":
        return PairGridConfig(
            upper=SectionConfig(continuous="corr_text", combo="box", discrete="count_heatmap"),
            lower=SectionConfig(continuous="scatter_reg", combo="box", discrete="stacked_bar"),
            diag=SectionConfig(continuous="hist_diag", discrete="bar_diag"),
        )
    raise ValueError(f"Unknown preset '{name}'. Available presets: default, minimal, kde, regression.")


def _merge_section(base: SectionConfig, override: dict[str, Any] | SectionConfig) -> SectionConfig:
    if isinstance(override, SectionConfig):
        return override
    if not isinstance(override, dict):
        raise TypeError("Section overrides must be dict or SectionConfig.")
    valid_keys = {"continuous", "combo", "discrete"}
    unknown = set(override.keys()) - valid_keys
    if unknown:
        unknown_joined = ", ".join(sorted(unknown))
        raise ValueError(f"Unknown section keys: {unknown_joined}")
    return replace(base, **override)


def merge_config(base: PairGridConfig, overrides: dict[str, Any]) -> PairGridConfig:
    if not overrides:
        return base
    nested_keys = {"upper", "lower", "diag"}
    merged: dict[str, Any] = {}
    for key, value in overrides.items():
        if key in nested_keys:
            merged[key] = _merge_section(getattr(base, key), value)
        else:
            merged[key] = value
    return replace(base, **merged)
