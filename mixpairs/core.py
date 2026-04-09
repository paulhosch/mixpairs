from __future__ import annotations

import logging
import warnings
from typing import Any

import numpy as np
import pandas as pd

from ._registry import resolve_renderer
from .config import PairGridConfig, SectionConfig, get_preset, merge_config
from .dispatch import build_execution_plan, validate_plan
from .layout import add_legend, configure_labels, create_axes, create_figure, finalize_figure
from .utils import classify_columns, compute_font_sizes, prepare_data, resolve_palette

logger = logging.getLogger(__name__)


def _section_from_user(value: dict[str, Any] | SectionConfig | None) -> dict[str, Any] | SectionConfig | None:
    if value is None:
        return None
    if isinstance(value, SectionConfig):
        return value
    if not isinstance(value, dict):
        raise TypeError("upper/lower/diag must be dict or SectionConfig.")
    return value


def ggpairs(
    data,
    columns=None,
    hue=None,
    palette=None,
    upper=None,
    lower=None,
    diag=None,
    corner=False,
    height=2.0,
    aspect=1.0,
    cardinality_threshold=15,
    column_labels=None,
    dtypes=None,
    sort_by_type=False,
    subsample=None,
    dropna_hue=True,
    title=None,
    legend_position="right",
    font_scale=1.0,
    preset="default",
    **kwargs,
):
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame.")
    cfg = get_preset(preset)
    overrides: dict[str, Any] = {
        "hue": hue,
        "palette": palette,
        "corner": corner,
        "height": height,
        "aspect": aspect,
        "cardinality_threshold": cardinality_threshold,
        "column_labels": column_labels,
        "dtypes": dtypes,
        "sort_by_type": sort_by_type,
        "subsample": subsample,
        "dropna_hue": dropna_hue,
        "title": title,
        "legend_position": legend_position,
        "font_scale": font_scale,
    }
    for section_name, section in (("upper", upper), ("lower", lower), ("diag", diag)):
        parsed = _section_from_user(section)
        if parsed is not None:
            overrides[section_name] = parsed
    overrides = {k: v for k, v in overrides.items() if v is not None}
    config: PairGridConfig = merge_config(cfg, overrides)

    if columns is None:
        columns = [c for c in data.columns if c != config.hue]
    else:
        columns = list(columns)
    if config.hue is not None:
        if config.hue not in data.columns:
            raise ValueError(f"hue column '{config.hue}' not found in data.")
        if config.hue in columns:
            warnings.warn(f"hue column '{config.hue}' removed from plotting columns.", stacklevel=2)
            columns = [c for c in columns if c != config.hue]
    missing = [c for c in columns if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found: {missing}")
    if not columns:
        raise ValueError("No columns selected for plotting.")

    prepared = prepare_data(data, columns, config.hue, config.subsample, config.dropna_hue)
    full_for_stats = prepare_data(data, columns, config.hue, None, config.dropna_hue)

    dtype_map = classify_columns(prepared, columns, config.cardinality_threshold, config.dtypes)
    if config.sort_by_type:
        cont = [c for c in columns if dtype_map[c].value == "continuous"]
        disc = [c for c in columns if dtype_map[c].value != "continuous"]
        columns = cont + disc

    plan = build_execution_plan(columns=columns, dtypes=dtype_map, config=config, registry_resolver=resolve_renderer)
    for warning_message in validate_plan(plan, prepared):
        logger.warning(warning_message)

    font_sizes = compute_font_sizes(len(columns), config.font_scale)
    fig, gs = create_figure(len(columns), config)
    axes = create_axes(fig, gs, plan, config, dtype_map, columns)
    palette_resolved = resolve_palette(config.palette, config.hue, prepared)
    render_failures: list[tuple[int, int, str, str | None, str]] = []

    for cell in plan:
        if cell.is_blank:
            continue
        ax = axes.get((cell.row, cell.col))
        if ax is None:
            continue
        try:
            if cell.y_col is None:
                cell.renderer(
                    ax=ax,
                    data=prepared,
                    col=cell.x_col,
                    hue_col=config.hue,
                    palette=palette_resolved,
                    **kwargs,
                )
            else:
                renderer_kwargs = dict(kwargs)
                if getattr(cell.renderer, "__name__", "") == "corr_text":
                    renderer_kwargs["full_data"] = full_for_stats
                cell.renderer(
                    ax=ax,
                    data=prepared,
                    x_col=cell.x_col,
                    y_col=cell.y_col,
                    hue_col=config.hue,
                    palette=palette_resolved,
                    orient=cell.orient,
                    **renderer_kwargs,
                )
        except Exception as exc:
            logger.warning(
                "Renderer failed at cell (%s, %s) for x='%s' y='%s': %s",
                cell.row,
                cell.col,
                cell.x_col,
                cell.y_col,
                exc,
            )
            render_failures.append((cell.row, cell.col, cell.x_col, cell.y_col, str(exc)))
            ax.text(0.5, 0.5, "X", transform=ax.transAxes, ha="center", va="center", color="red", fontsize=16)

    configure_labels(axes, columns, config, font_sizes)
    add_legend(fig, axes, config.hue, palette_resolved, config, font_sizes)
    finalize_figure(fig, config, font_sizes)

    if render_failures:
        sample = "; ".join(
            [f"({r},{c}) x={x}, y={y}, err={err}" for r, c, x, y, err in render_failures[:3]]
        )
        warnings.warn(
            f"{len(render_failures)} renderer cell(s) failed. Sample: {sample}",
            stacklevel=2,
        )

    n = len(columns)
    axes_array = np.empty((n, n), dtype=object)
    axes_array[:] = None
    for (r, c), ax in axes.items():
        axes_array[r, c] = ax
    return fig, axes_array
