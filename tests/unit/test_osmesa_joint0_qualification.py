from pathlib import Path

import numpy as np
import pytest

from epsbench.diagnostics.osmesa_joint0_qualification import (
    QualificationFailure,
    RendererAdapter,
    compare_repeat_artifacts,
    fixed_attempts,
    plan,
    reserve,
    validate_plan,
)


class Quality:
    offsamples = 4


class Vis:
    quality = Quality()


class Model:
    vis = Vis()


class Fake:
    def __init__(self):
        self.Renderer = self.original

    def original(self, model, *a, **k):
        return object()


def facts(renderer):
    return {"offsamples": 0, "sample_count": 0, "osmesa_context": True, "gl_error": 0}


def test_fixed_budget():
    p = plan()
    assert len(fixed_attempts()) == 8
    assert p["limits"]["native_render_readbacks"] == 224
    assert [a.expected_calls for a in fixed_attempts()] == [8, 8, 8, 8, 6, 6, 6, 6]


def test_adapter_sets_and_restores():
    m = Fake()
    model = Model()
    with RendererAdapter(m, facts) as hook:
        m.Renderer(model)
        assert model.vis.quality.offsamples == 0
    assert m.Renderer.__func__ is hook.original.__func__


def test_adapter_double_and_bad_facts_restore():
    m = Fake()
    model = Model()
    with pytest.raises(QualificationFailure):
        with RendererAdapter(m, facts):
            m.Renderer(model)
            m.Renderer(model)
    assert m.Renderer.__func__ is m.original.__func__
    m = Fake()
    with pytest.raises(QualificationFailure):
        with RendererAdapter(m, lambda r: {"offsamples": 1}):
            m.Renderer(Model())
    assert m.Renderer.__func__ is m.original.__func__


def test_ledger_terminal(tmp_path: Path):
    a = fixed_attempts()[0]
    reserve(tmp_path, a, {"x": 1})
    with pytest.raises(QualificationFailure):
        reserve(tmp_path, fixed_attempts()[1], {"x": 1})


def test_exact_repeat_falsifies_array(tmp_path: Path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    np.save(a / "x.npy", np.array([1], dtype=np.int32))
    np.save(b / "x.npy", np.array([2], dtype=np.int32))
    with pytest.raises(QualificationFailure):
        compare_repeat_artifacts(a, b)


def test_plan_mutation_rejected():
    q = plan()
    q["seed"] = 1
    with pytest.raises(QualificationFailure):
        validate_plan(q)
