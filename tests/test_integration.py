import matplotlib

matplotlib.use("Agg")
import matplotlib.figure
import numpy as np
import pandas as pd
import pytest

from mixpairs import ggpairs


def _mixed_df(n=120):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "a": rng.normal(0, 1, n),
            "b": rng.normal(1, 2, n),
            "c": rng.normal(-1, 0.5, n),
            "d": rng.choice(["x", "y", "z"], size=n),
            "hue": rng.choice(["g1", "g2"], size=n),
        }
    )


def test_ggpairs_iris_returns_figure_and_array():
    df = _mixed_df(150)
    fig, axes = ggpairs(df, columns=["a", "b", "c", "d"], hue="hue")
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (4, 4)
    texts = [t.get_text() for ax in axes.flat if ax is not None for t in ax.texts]
    assert "X" not in texts


def test_ggpairs_tips_corner_mode():
    df = _mixed_df(100)
    fig, axes = ggpairs(df, columns=["a", "b", "d"], corner=True)
    assert isinstance(fig, matplotlib.figure.Figure)
    assert axes.shape == (3, 3)
    assert axes[0, 1] is None
    assert axes[0, 2] is None


def test_ggpairs_custom_renderer_callable():
    calls = {"n": 0}

    def custom_scatter(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
        calls["n"] += 1
        ax.scatter(data[x_col], data[y_col], s=5)

    df = _mixed_df(80)
    ggpairs(df, columns=["a", "b", "c"], upper={"continuous": custom_scatter})
    assert calls["n"] > 0


def test_ggpairs_warns_when_renderer_fails():
    def bad_renderer(ax, data, x_col, y_col, hue_col, palette, orient, **kwargs):
        raise RuntimeError("forced renderer failure")

    df = _mixed_df(60)
    with pytest.warns(UserWarning, match="renderer cell\\(s\\) failed"):
        _, axes = ggpairs(df, columns=["a", "b", "c"], upper={"continuous": bad_renderer})
    texts = [t.get_text() for ax in axes.flat if ax is not None for t in ax.texts]
    assert "X" in texts
