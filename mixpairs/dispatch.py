from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from .config import DType, PairGridConfig, PairType, Triangle
from .utils import build_pair_type_matrix, determine_orient


@dataclass(frozen=True)
class CellPlan:
    row: int
    col: int
    triangle: Triangle
    pair_type: PairType
    renderer: Callable[..., Any]
    x_col: str
    y_col: str | None
    orient: Any | None
    is_blank: bool


def _triangle(i: int, j: int) -> Triangle:
    if i < j:
        return Triangle.UPPER
    if i > j:
        return Triangle.LOWER
    return Triangle.DIAG


def _renderer_key(config: PairGridConfig, triangle: Triangle, pair_type: PairType) -> tuple[Any, str]:
    section = (
        config.upper
        if triangle == Triangle.UPPER
        else config.lower
        if triangle == Triangle.LOWER
        else config.diag
    )
    if pair_type == PairType.CONTINUOUS:
        return section.continuous, "bivar"
    if pair_type == PairType.COMBO:
        return section.combo, "bivar"
    if pair_type == PairType.DISCRETE:
        return section.discrete, "bivar"
    if pair_type == PairType.DIAG_CONTINUOUS:
        return section.continuous, "univar"
    if pair_type == PairType.DIAG_CATEGORICAL:
        return section.discrete, "univar"
    raise ValueError(f"Unsupported PairType: {pair_type}")


def build_execution_plan(
    columns: list[str],
    dtypes: dict[str, DType],
    config: PairGridConfig,
    registry_resolver: Callable[[Any, str], Callable[..., Any]],
) -> list[CellPlan]:
    matrix = build_pair_type_matrix(columns, dtypes)
    plan: list[CellPlan] = []
    for i, row_name in enumerate(columns):
        for j, col_name in enumerate(columns):
            triangle = _triangle(i, j)
            corner_blank = config.corner and triangle == Triangle.UPPER
            pair_type = matrix[i][j]
            key, kind = _renderer_key(config, triangle, pair_type)
            renderer = registry_resolver("blank" if corner_blank else key, kind)
            orient = None
            if pair_type == PairType.COMBO:
                orient = determine_orient(dtypes[row_name], dtypes[col_name], triangle)
            plan.append(
                CellPlan(
                    row=i,
                    col=j,
                    triangle=triangle,
                    pair_type=pair_type,
                    renderer=renderer,
                    x_col=col_name,
                    y_col=None if triangle == Triangle.DIAG else row_name,
                    orient=orient,
                    is_blank=corner_blank or key in {None, "blank"},
                )
            )
    return plan


def validate_plan(plan: list[CellPlan], data: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    for cell in plan:
        if cell.is_blank:
            continue
        if getattr(cell.renderer, "__name__", "") == "hexbin":
            warnings.append(
                f"Cell ({cell.row}, {cell.col}) uses hexbin. If hue is set it will fallback to scatter."
            )
        if cell.pair_type == PairType.COMBO and cell.x_col in data.columns and cell.y_col in data.columns:
            x_levels = data[cell.x_col].nunique(dropna=True)
            y_levels = data[cell.y_col].nunique(dropna=True) if cell.y_col else 0
            if max(x_levels, y_levels) >= 50:
                warnings.append(
                    f"Cell ({cell.row}, {cell.col}) has very high categorical cardinality and may