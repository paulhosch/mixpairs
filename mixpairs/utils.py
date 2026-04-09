from __future__ import annotations

from typing import Any

import pandas as pd
import seaborn as sns
from pandas.api.types import (
    CategoricalDtype,
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)

from .config import DType, Orient, PairType, Triangle


def classify_column(series: pd.Series, threshold: int, forced: DType | None) -> DType:
    if forced is not None:
        return forced
    if (
        isinstance(series.dtype, CategoricalDtype)
        or is_object_dtype(series.dtype)
        or is_string_dtype(series.dtype)
        or is_bool_dtype(series.dtype)
    ):
        return DType.CATEGORICAL
    if is_numeric_dtype(series.dtype):
        unique_n = series.nunique(dropna=True)
        if unique_n < threshold:
            return DType.CATEGORICAL
        return DType.CONTINUOUS
    return DType.CATEGORICAL


def classify_columns(
    df: pd.DataFrame,
    columns: list[str],
    threshold: int,
    forced: dict[str, DType] | None,
) -> dict[str, DType]:
    forced = forced or {}
    return {col: classify_column(df[col], threshold, forced.get(col)) for col in columns}


def determine_pair_type(dtype_i: DType, dtype_j: DType) -> PairType:
    i_is_cont = dtype_i == DType.CONTINUOUS
    j_is_cont = dtype_j == DType.CONTINUOUS
    if i_is_cont and j_is_cont:
        return PairType.CONTINUOUS
    if not i_is_cont and not j_is_cont:
        return PairType.DISCRETE
    return PairType.COMBO


def determine_orient(dtype_row: DType, dtype_col: DType, triangle: Triangle) -> Orient:
    if triangle == Triangle.LOWER:
        return Orient.VERTICAL if dtype_col != DType.CONTINUOUS else Orient.HORIZONTAL
    if triangle == Triangle.UPPER:
        return Orient.VERTICAL if dtype_row != DType.CONTINUOUS else Orient.HORIZONTAL
    return Orient.VERTICAL


def build_pair_type_matrix(columns: list[str], dtypes: dict[str, DType]) -> list[list[PairType]]:
    matrix: list[list[PairType]] = []
    for row_col in columns:
        row: list[PairType] = []
        for col_col in columns:
            if row_col == col_col:
                row.append(
                    PairType.DIAG_CONTINUOUS
                    if dtypes[row_col] == DType.CONTINUOUS
                    else PairType.DIAG_CATEGORICAL
                )
            else:
                row.append(determine_pair_type(dtypes[row_col], dtypes[col_col]))
        matrix.append(row)
    return matrix


def compute_font_sizes(n_cols: int, font_scale: float) -> dict[str, float]:
    base = {
        "title": 14.0,
        "axis_label": 10.0,
        "tick_label": 8.0,
        "corr_text": 12.0,
        "legend": 10.0,
    }
    if n_cols <= 0:
        scale = 1.0
    else:
        scale = max(0.5, min(1.5, 4.0 / n_cols))
    scale *= font_scale
    return {k: v * scale for k, v in base.items()}


def resolve_palette(palette: Any, hue_col: str | None, data: pd.DataFrame) -> dict | list | None:
    if hue_col is None:
        return palette
    levels = sorted(data[hue_col].dropna().unique().tolist())
    if palette is None:
        colors = sns.color_palette(n_colors=len(levels))
        return dict(zip(levels, colors))
    if isinstance(palette, str):
        colors = sns.color_palette(palette, n_colors=len(levels))
        return dict(zip(levels, colors))
    if isinstance(palette, dict):
        missing = [level for level in levels if level not in palette]
        if missing:
            raise ValueError(f"Palette dict missing hue levels: {missing}")
        return palette
    if isinstance(palette, list):
        if len(palette) != len(levels):
            raise ValueError(
                f"Palette list has {len(palette)} colors but hue '{hue_col}' has {len(levels)} levels."
            )
        return dict(zip(levels, palette))
    raise TypeError("palette must be None, string, list, or dict.")


def prepare_data(
    data: pd.DataFrame,
    columns: list[str],
    hue: str | None,
    subsample: int | None,
    dropna_hue: bool,
) -> pd.DataFrame:
    keep_cols = list(columns)
    if hue is not None and hue not in keep_cols:
        keep_cols.append(hue)
    df = data.loc[:, keep_cols].copy()
    if hue is not None and dropna_hue:
        df = df[df[hue].notna()]
    if subsample is not None:
        if subsample <= 0:
            raise ValueError("subsample must be > 0.")
        if len(df) > subsample:
            df = df.sample(n=subsample, random_state=0)
    return df
