import pytest

from mixpairs.config import PairGridConfig, SectionConfig, get_preset, merge_config


def test_get_preset_default():
    cfg = get_preset("default")
    assert isinstance(cfg, PairGridConfig)
    assert cfg.upper.continuous == "corr_text"


def test_get_preset_unknown():
    with pytest.raises(ValueError):
        get_preset("unknown")


def test_merge_config_nested_sections():
    base = get_preset("default")
    merged = merge_config(base, {"upper": {"continuous": "kde2d"}, "font_scale": 1.2})
    assert merged.upper.continuous == "kde2d"
    assert merged.upper.combo == base.upper.combo
    assert merged.font_scale == 1.2


def test_section_config_defaults():
    sec = SectionConfig()
    assert sec.continuous is None
    assert sec.combo is None
    assert sec.discrete is None
