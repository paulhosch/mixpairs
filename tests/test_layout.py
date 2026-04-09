import matplotlib

matplotlib.use("Agg")

from mixpairs._registry import resolve_renderer
from mixpairs.config import DType, get_preset
from mixpairs.dispatch import build_execution_plan
from mixpairs.layout import configure_labels, create_axes, create_figure
from mixpairs.utils import compute_font_sizes


def _setup(corner=False):
    columns = ["a", "b", "c"]
    dtypes = {"a": DType.CONTINUOUS, "b": DType.CONTINUOUS, "c": DType.CATEGORICAL}
    cfg = get_preset("default")
    cfg = cfg.__class__(**{**cfg.__dict__, "corner": corner})
    plan = build_execution_plan(columns, dtypes, cfg, resolve_renderer)
    fig, gs = create_figure(len(columns), cfg)
    axes = create_axes(fig, gs, plan, cfg, dtypes, columns)
    return fig, axes, columns, cfg


def test_create_figure_dimensions():
    cfg = get_preset("default")
    fig, _ = create_figure(3, cfg)
    w, h = fig.get_size_inches()
    assert w > 0
    assert h > 0


def test_create_axes_count_for_full_and_corner():
    _, axes_full, _, _ = _setup(corner=False)
    _, axes_corner, _, _ = _setup(corner=True)
    assert len(axes_full) == 9
    assert len(axes_corner) == 6


def test_configure_labels_edges():
    _, axes, columns, cfg = _setup(corner=False)
    font_sizes = compute_font_sizes(len(columns), 1.0)
    configure_labels(axes, columns, cfg, font_sizes)
    assert axes[(2, 2)].get_xlabel() != "