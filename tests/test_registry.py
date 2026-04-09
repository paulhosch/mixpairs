import pytest

from mixpairs._registry import _BIVAR_REGISTRY, _UNIVAR_REGISTRY, resolve_renderer


def test_builtins_registered():
    assert "scatter" in _BIVAR_REGISTRY
    assert "corr_text" in _BIVAR_REGISTRY
    assert "kde_diag" in _UNIVAR_REGISTRY


def test_resolve_renderer_valid_invalid():
    fn = resolve_renderer("scatter", "bivar")
    assert callable(fn)
    with pytest.raises(KeyError):
        resolve_renderer("not_exists", "bivar")


def test_resolve_renderer_callable_signature_validation():
    def bad(ax, data, x_col, y_col, hue_col, palette, orient):
        pass

    with pytest.raises(TypeError):
        resolve_renderer(bad, "bivar")
