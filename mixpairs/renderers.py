from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from .config import Orient


def _remove_axis_legend(ax) -> None:
    legend = ax.get_legend()
    if legend is not None:
        legend.remove()


def _draw_empty(ax) -> None:
    ax.text(0.5, 0.5, "n=0", transform=ax.transAxes, ha="center", va="center", fontsize=9, alpha=0.6)


def _drop_xy(data: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    return data[[x_col, y_col] + [c for c in data.columns if c not in {x_col, y_col}]].dropna(
        subset=[x_col, y_col]
    )


def _drop_x(data: pd.DataFrame, col: str) -> pd.DataFrame:
    return data[[col] + [c for c in data.columns if c != col]].dropna(subset=[col])


def _combo_columns(x_col: str, y_col: str, orient: Orient) -> tuple[str, str]:
    if orient == Orient.VERTICAL:
        return x_col, y_col
    return y_col, x_col


def _p_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def scatter(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    default = {"alpha": 0.5, "s": 10}
    default.update(kwargs)
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax, **default)
    _remove_axis_legend(ax)


def scatter_reg(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    if hue_col is None:
        sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter_kws={"alpha": kwargs.pop("alpha", 0.5), "s": kwargs.pop("s", 10)}, **kwargs)
        _remove_axis_legend(ax)
        return
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax, alpha=kwargs.get("alpha", 0.5), s=kwargs.get("s", 10))
    levels = sorted(df[hue_col].dropna().unique().tolist())
    palette_map = palette if isinstance(palette, dict) else dict(zip(levels, sns.color_palette(n_colors=len(levels))))
    for level in levels:
        grp = df[df[hue_col] == level]
        if grp.shape[0] < 2:
            continue
        sns.regplot(data=grp, x=x_col, y=y_col, ax=ax, scatter=False, color=palette_map[level], line_kws={"linewidth": 1.5})
    _remove_axis_legend(ax)


def kde2d(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    default = {"fill": True, "alpha": 0.3, "levels": 5}
    default.update(kwargs)
    sns.kdeplot(data=df, x=x_col, y=y_col, hue=hue_col, palette=palette, ax=ax, **default)
    _remove_axis_legend(ax)


def hexbin(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    if hue_col is not None:
        warnings.warn("hexbin does not support hue; falling back to scatter.", stacklevel=2)
        scatter(ax=ax, data=df, x_col=x_col, y_col=y_col, hue_col=hue_col, palette=palette, orient=orient, **kwargs)
        return
    gridsize = kwargs.pop("gridsize", 25)
    cmap = kwargs.pop("cmap", "viridis")
    ax.hexbin(df[x_col], df[y_col], gridsize=gridsize, cmap=cmap, mincnt=1, **kwargs)
    _remove_axis_legend(ax)


def corr_text(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    corr_data = kwargs.pop("full_data", data)
    df = _drop_xy(corr_data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    method = kwargs.pop("method", "pearson")
    shade = kwargs.pop("shade", True)

    def _corr(a: pd.Series, b: pd.Series) -> tuple[float, float]:
        if method == "spearman":
            r, p = spearmanr(a, b)
        else:
            r, p = pearsonr(a, b)
        return float(r), float(p)

    ax.set_xticks([])
    ax.set_yticks([])
    if hue_col is None:
        r, p = _corr(df[x_col], df[y_col])
        txt = f"r={r:.2f}{_p_stars(p)}"
        size = 10 + 10 * abs(r)
        ax.text(0.5, 0.5, txt, transform=ax.transAxes, ha="center", va="center", fontsize=size, weight="bold")
        if shade:
            cmap = kwargs.pop("cmap", "RdBu_r")
            norm = (r + 1) / 2
            ax.set_facecolor(plt_colormap(cmap, norm))
    else:
        levels = sorted(df[hue_col].dropna().unique().tolist())
        palette_map = palette if isinstance(palette, dict) else dict(zip(levels, sns.color_palette(n_colors=len(levels))))
        y0 = 0.85
        dy = 0.7 / max(len(levels), 1)
        for idx, level in enumerate(levels):
            grp = df[df[hue_col] == level]
            if grp.shape[0] < 2:
                continue
            r, p = _corr(grp[x_col], grp[y_col])
            ax.text(
                0.5,
                y0 - idx * dy,
                f"{level}: {r:.2f}{_p_stars(p)}",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9,
                color=palette_map.get(level),
            )
    _remove_axis_legend(ax)


def box(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    cat_col, cont_col = _combo_columns(x_col, y_col, orient)
    local_hue = None if hue_col == cat_col else hue_col
    if orient == Orient.VERTICAL:
        sns.boxplot(data=df, x=cat_col, y=cont_col, hue=local_hue, palette=palette, ax=ax, **kwargs)
    else:
        sns.boxplot(data=df, y=cat_col, x=cont_col, hue=local_hue, palette=palette, ax=ax, **kwargs)
    _remove_axis_legend(ax)


def violin(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    cat_col, cont_col = _combo_columns(x_col, y_col, orient)
    local_hue = None if hue_col == cat_col else hue_col
    if orient == Orient.VERTICAL:
        sns.violinplot(data=df, x=cat_col, y=cont_col, hue=local_hue, palette=palette, ax=ax, **kwargs)
    else:
        sns.violinplot(data=df, y=cat_col, x=cont_col, hue=local_hue, palette=palette, ax=ax, **kwargs)
    _remove_axis_legend(ax)


def strip(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    cat_col, cont_col = _combo_columns(x_col, y_col, orient)
    local_hue = None if hue_col == cat_col else hue_col
    default = {"jitter": True, "alpha": 0.7, "size": 3}
    default.update(kwargs)
    if orient == Orient.VERTICAL:
        sns.stripplot(data=df, x=cat_col, y=cont_col, hue=local_hue, palette=palette, ax=ax, **default)
    else:
        sns.stripplot(data=df, y=cat_col, x=cont_col, hue=local_hue, palette=palette, ax=ax, **default)
    _remove_axis_legend(ax)


def facet_hist(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    cat_col, cont_col = _combo_columns(x_col, y_col, orient)
    levels = sorted(df[cat_col].dropna().unique().tolist())
    colors = sns.color_palette(n_colors=len(levels))
    for idx, level in enumerate(levels):
        grp = df[df[cat_col] == level]
        sns.histplot(grp[cont_col], ax=ax, stat="count", element="step", fill=True, alpha=0.35, color=colors[idx], label=str(level), **kwargs)
    _remove_axis_legend(ax)


def facet_kde(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    cat_col, cont_col = _combo_columns(x_col, y_col, orient)
    levels = sorted(df[cat_col].dropna().unique().tolist())
    colors = sns.color_palette(n_colors=len(levels))
    for idx, level in enumerate(levels):
        grp = df[df[cat_col] == level]
        if grp.shape[0] < 2:
            continue
        sns.kdeplot(grp[cont_col], ax=ax, color=colors[idx], label=str(level), **kwargs)
    _remove_axis_legend(ax)


def count_heatmap(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    table = pd.crosstab(df[y_col], df[x_col])
    sns.heatmap(table, ax=ax, annot=True, fmt="d", cbar=False, cmap=kwargs.pop("cmap", "Blues"), **kwargs)
    _remove_axis_legend(ax)


def stacked_bar(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
    df = _drop_xy(data, x_col, y_col)
    if df.empty:
        _draw_empty(ax)
        return
    table = pd.crosstab(df[x_col], df[y_col])
    table.plot(kind="bar", stacked=True, ax=ax, legend=False, **kwargs)
    _remove_axis_legend(ax)


def hist_diag(ax, data, col, hue_col, palette, **kwargs):
    df = _drop_x(data, col)
    if df.empty:
        _draw_empty(ax)
        return
    sns.histplot(data=df, x=col, hue=hue_col, palette=palette, kde=False, ax=ax, alpha=0.5, **kwargs)
    _remove_axis_legend(ax)


def kde_diag(ax, data, col, hue_col, palette, **kwargs):
    df = _drop_x(data, col)
    if df.empty:
        _draw_empty(ax)
        return
    sns.kdeplot(data=df, x=col, hue=hue_col, palette=palette, ax=ax, fill=False, **kwargs)
    _remove_axis_legend(ax)


def bar_diag(ax, data, col, hue_col, palette, **kwargs):
    df = _drop_x(data, col)
    if df.empty:
        _draw_empty(ax)
        return
    sns.countplot(data=df, x=col, hue=hue_col, palette=palette, ax=ax, **kwargs)
    _remove_axis_legend(ax)


def _blank_renderer(ax, *args, **kwargs):
    ax.set_visible(False)


def plt_colormap(name: str, value: float) -> tuple[float, float, float, float]:
    from matplotlib import colormaps

    cmap = colormaps[name]
    return cmap(min(1.0, max(0.0, value)))
