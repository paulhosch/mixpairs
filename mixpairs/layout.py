from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from .config import DType, PairGridConfig
from .dispatch import CellPlan


def create_figure(n_cols: int, config: PairGridConfig) -> tuple[Figure, GridSpec]:
    base_width = n_cols * config.height * config.aspect
    base_height = n_cols * config.height
    n_rows = n_cols
    n_gs_cols = n_cols
    width_ratios = [1.0] * n_cols
    height_ratios = [1.0] * n_cols
    if config.legend_position == "right":
        n_gs_cols += 1
        width_ratios.append(0.25)
        base_width += config.height * 0.8
    elif config.legend_position == "bottom":
        n_rows += 1
        height_ratios.append(0.25)
        base_height += config.height * 0.8

    fig = plt.figure(figsize=(base_width, base_height))
    gs = fig.add_gridspec(n_rows, n_gs_cols, width_ratios=width_ratios, height_ratios=height_ratios)
    return fig, gs


def create_axes(
    fig: Figure,
    gs: GridSpec,
    plan: list[CellPlan],
    config: PairGridConfig,
    dtypes: dict[str, DType],
    columns: list[str],
) -> dict[tuple[int, int], Any]:
    axes: dict[tuple[int, int], Any] = {}
    share_x: dict[int, Any] = {}
    share_y: dict[int, Any] = {}
    col_by_idx = {i: col for i, col in enumerate(columns)}

    for cell in plan:
        if cell.is_blank:
            continue
        row, col = cell.row, cell.col
        sharex = None
        sharey = None
        if config.share_axes and dtypes[col_by_idx[col]] == DType.CONTINUOUS:
            sharex = share_x.get(col)
        if (
            config.share_axes
            and cell.row != cell.col
            and dtypes[col_by_idx[row]] == DType.CONTINUOUS
        ):
            sharey = share_y.get(row)
        ax = fig.add_subplot(gs[row, col], sharex=sharex, sharey=sharey)
        axes[(row, col)] = ax
        share_x.setdefault(col, ax)
        if row != col:
            share_y.setdefault(row, ax)
    return axes


def configure_labels(axes, columns, config, font_sizes):
    n = len(columns)
    label_map = config.column_labels or {}
    for (row, col), ax in axes.items():
        is_bottom = row == n - 1
        is_left = col == 0
        if config.corner:
            below = [r for (r, c) in axes.keys() if c == col]
            is_bottom = row == max(below)
        ax.tick_params(labelbottom=is_bottom, labelleft=is_left, labelsize=font_sizes["tick_label"])
        if is_bottom:
            ax.set_xlabel(label_map.get(columns[col], columns[col]), fontsize=font_sizes["axis_label"])
            if n > 6:
                for tick in ax.get_xticklabels():
                    tick.set_rotation(45)
                    tick.set_ha("right")
        else:
            ax.set_xlabel("")
        if is_left:
            ax.set_ylabel(label_map.get(columns[row], columns[row]), fontsize=font_sizes["axis_label"])
        else:
            ax.set_ylabel("")


def add_legend(fig, axes, hue_col, palette, config, font_sizes):
    if hue_col is None or config.legend_position == "none":
        return
    handles = None
    labels = None
    for ax in axes.values():
        h, l = ax.get_legend_handles_labels()
        if h and l:
            handles, labels = h, l
            break
    if handles is None and isinstance(palette, dict):
        labels = [str(k) for k in palette.keys()]
        handles = [Line2D([0], [0], marker="o", linestyle="", color=v) for v in palette.values()]
    if not handles:
        return
    if config.legend_position == "right":
        fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=font_sizes["legend"])
    elif config.legend_position == "bottom":
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=max(1, len(labels)), fontsize=font_sizes["legend"])


def finalize_figure(fig, config, font_sizes):
    if config.title:
        fig.suptitle(config.title, fontsize=font_sizes["title"])
    fig.tight_layout(pad=0.8)
    if config.legend_position == "right":
        fig.subplots_adjust(right=0.9)
    elif config.legend_position == "bottom":
        fig.subplots_adjust(bottom=0.12)
