import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from mixpairs.config import Orient
from mixpairs.renderers import (
    bar_diag,
    box,
    corr_text,
    count_heatmap,
    facet_hist,
    facet_kde,
    hexbin,
    hist_diag,
    kde2d,
    kde_diag,
    scatter,
    scatter_reg,
    stacked_bar,
    strip,
    violin,
)


def _df():
    return pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1.2, 1.9, 2.8, 4.1, 5.2, 6.0],
            "cat": ["a", "a", "b", "b", "c", "c"],
            "hue": ["h1", "h2", "h1", "h2", "h1", "h2"],
        }
    )


@pytest.mark.parametrize("fn", [scatter, scatter_reg, kde2d, hexbin, corr_text])
def test_continuous_renderers(fn):
    fig, ax = plt.subplots()
    fn(ax, _df(), "x", "y", "hue", None, Orient.VERTICAL)
    assert len(ax.get_children()) > 0
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.parametrize("fn", [box, violin, strip, facet_hist, facet_kde])
def test_combo_renderers(fn):
    fig, ax = plt.subplots()
    fn(ax, _df(), "cat", "y", "hue", None, Orient.VERTICAL)
    assert len(ax.get_children()) > 0
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.parametrize("fn", [count_heatmap, stacked_bar])
def test_discrete_renderers(fn):
    fig, ax = plt.subplots()
    fn(ax, _df(), "cat", "hue", None, None, Orient.VERTICAL)
    assert len(ax.get_children()) > 0
    assert ax.get_legend() is None
    plt.close(fig)


@pytest.mark.parametrize("fn", [hist_diag, kde_diag, bar_diag])
def test_diag_renderers(fn):
    fig, ax = plt.subplots()
    fn(ax, _df(), "x" if fn != bar_diag else "cat", "hue", None)
    assert len(ax.get_children()) > 0
    assert ax.get_legend() is None
    plt.close(fig)


def test_empty_data_no_exception():
    fig, ax = plt.subplots()
    empty = _df().iloc[0:0]
    scatter(ax, empty, "x", "y", "hue", None, Orient.VERTICAL)
    assert len(ax.texts) >= 1
    plt.close(fig)
