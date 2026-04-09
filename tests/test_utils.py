import pandas as pd

from mixpairs.config import DType, PairType, Triangle
from mixpairs.utils import (
    build_pair_type_matrix,
    classify_column,
    classify_columns,
    compute_font_sizes,
    determine_orient,
    prepare_data,
)


def test_classify_column_cases():
    assert classify_column(pd.Series([1.1, 2.2, 3.3, 4.4]), threshold=3, forced=None) == DType.CONTINUOUS
    assert classify_column(pd.Series(["a", "b", "a"]), threshold=15, forced=None) == DType.CATEGORICAL
    assert classify_column(pd.Series([1, 1, 2, 2, 3]), threshold=15, forced=None) == DType.CATEGORICAL
    assert classify_column(pd.Series(range(50)), threshold=15, forced=None) == DType.CONTINUOUS
    assert classify_column(pd.Series([True, False, True]), threshold=15, forced=None) == DType.CATEGORICAL
    assert classify_column(pd.Series([1, 2, 3]), threshold=15, forced=DType.ORDINAL) == DType.ORDINAL


def test_classify_columns():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "z"]})
    out = classify_columns(df, ["a", "b"], threshold=3, forced=None)
    assert out["a"] == DType.CONTINUOUS
    assert out["b"] == DType.CATEGORICAL


def test_determine_orient_variants():
    assert determine_orient(DType.CATEGORICAL, DType.CONTINUOUS, Triangle.LOWER).value == "horizontal"
    assert determine_orient(DType.CONTINUOUS, DType.CATEGORICAL, Triangle.LOWER).value == "vertical"
    assert determine_orient(DType.CATEGORICAL, DType.CONTINUOUS, Triangle.UPPER).value == "vertical"


def test_build_pair_type_matrix():
    cols = ["x", "y", "z"]
    dtypes = {"x": DType.CONTINUOUS, "y": DType.CATEGORICAL, "z": DType.CONTINUOUS}
    matrix = build_pair_type_matrix(cols, dtypes)
    assert matrix[0][0] == PairType.DIAG_CONTINUOUS
    assert matrix[1][1] == PairType.DIAG_CATEGORICAL
    assert matrix[0][2] == PairType.CONTINUOUS
    assert matrix[1][0] == PairType.COMBO


def test_compute_font_sizes():
    small = compute_font_sizes(4, 1.0)
    large = compute_font_sizes(12, 1.0)
    assert set(small.keys()) == {"title", "axis_label", "tick_label", "corr_text", "legend"}
    assert large["axis_label"] < small["axis_label"]


def test_prepare_data():
    df = pd.DataFrame({"x": [1, 2, 3, 4], "h": ["a", None, "b", "b"]})
    out = prepare_data(df, columns=["x"], hue="h", subsample=2, dropna_hue=True)
    assert "x" in out.columns and "h" in out.columns
    assert out["h"].isna().sum() == 0
    assert len(out) <= 2
