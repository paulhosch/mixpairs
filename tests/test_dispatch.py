from mixpairs._registry import resolve_renderer
from mixpairs.config import DType, PairType, get_preset
from mixpairs.dispatch import build_execution_plan


def test_build_execution_plan_continuous():
    columns = ["a", "b", "c"]
    dtypes = {c: DType.CONTINUOUS for c in columns}
    cfg = get_preset("default")
    plan = build_execution_plan(columns, dtypes, cfg, resolve_renderer)
    assert len(plan) == 9
    diag = [p for p in plan if p.row == p.col]
    assert all(p.pair_type == PairType.DIAG_CONTINUOUS for p in diag)


def test_build_execution_plan_mixed_and_corner():
    columns = ["a", "b", "c"]
    dtypes = {"a": DType.CONTINUOUS, "b": DType.CONTINUOUS, "c": DType.CATEGORICAL}
    cfg = get_preset("default")
    cfg = cfg.__class__(**{**cfg.__dict__, "corner": True})
    plan = build_execution_plan(columns, dtypes, cfg, resolve_renderer)
    upper = [p for p in plan if p.row < p.col]
    assert all(p.is_blank for p in upper)
    combo = [p for p in plan if p.pair_type == PairType.COMBO]
    assert len(combo) > 0
