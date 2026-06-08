"""Tests for agent_cost_tracker."""

from __future__ import annotations

import pytest

from agent_cost_tracker import AgentCostTracker, ModelPrice, UnknownModelError

# We use a simple test model for deterministic cost math.
_MODEL = "test-model"
_INP_RATE = 2.0  # $2 per MTok input
_OUT_RATE = 8.0  # $8 per MTok output


def make_tracker() -> AgentCostTracker:
    t = AgentCostTracker()
    t.register(_MODEL, input_per_mtok=_INP_RATE, output_per_mtok=_OUT_RATE)
    return t


# ---------------------------------------------------------------------------
# ModelPrice
# ---------------------------------------------------------------------------


def test_model_price_cost_zero():
    mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
    assert mp.cost(0, 0) == pytest.approx(0.0)


def test_model_price_cost_input_only():
    mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
    # 1_000_000 input tokens at $2/MTok = $2
    assert mp.cost(1_000_000, 0) == pytest.approx(2.0)


def test_model_price_cost_output_only():
    mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
    assert mp.cost(0, 1_000_000) == pytest.approx(8.0)


def test_model_price_cost_combined():
    mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
    # 500k input + 250k output
    expected = 500_000 * 2.0 / 1_000_000 + 250_000 * 8.0 / 1_000_000
    assert mp.cost(500_000, 250_000) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Constructor / repr
# ---------------------------------------------------------------------------


def test_repr():
    t = AgentCostTracker()
    assert "calls=0" in repr(t)
    assert "total_cost_usd=0.000000" in repr(t)


def test_initial_empty():
    t = AgentCostTracker()
    assert t.call_count() == 0
    assert t.total_cost() == pytest.approx(0.0)
    assert t.total_tokens() == 0


# ---------------------------------------------------------------------------
# register / price / known_models / has_price
# ---------------------------------------------------------------------------


def test_register_returns_self():
    t = AgentCostTracker()
    assert t.register("m", input_per_mtok=1.0, output_per_mtok=2.0) is t


def test_register_overrides():
    t = AgentCostTracker()
    t.register("m", input_per_mtok=1.0, output_per_mtok=2.0)
    t.register("m", input_per_mtok=5.0, output_per_mtok=10.0)
    assert t.price("m").input_per_mtok == pytest.approx(5.0)


def test_has_price_true():
    t = make_tracker()
    assert t.has_price(_MODEL) is True


def test_has_price_false():
    assert AgentCostTracker().has_price("nope") is False


def test_known_models_sorted():
    t = AgentCostTracker()
    t.register("z", input_per_mtok=1, output_per_mtok=1)
    t.register("a", input_per_mtok=1, output_per_mtok=1)
    models = t.known_models()
    assert models.index("a") < models.index("z")


def test_price_unknown_raises():
    with pytest.raises(UnknownModelError):
        AgentCostTracker().price("nope")


def test_builtin_prices_loaded():
    t = AgentCostTracker()
    assert t.has_price("claude-sonnet-4-6")
    assert t.has_price("gpt-4o")


def test_register_negative_input_price_raises():
    t = AgentCostTracker()
    with pytest.raises(ValueError):
        t.register("m", input_per_mtok=-1.0, output_per_mtok=2.0)


def test_register_negative_output_price_raises():
    t = AgentCostTracker()
    with pytest.raises(ValueError):
        t.register("m", input_per_mtok=1.0, output_per_mtok=-2.0)


def test_model_price_negative_raises():
    with pytest.raises(ValueError):
        ModelPrice("m", input_per_mtok=-1.0, output_per_mtok=1.0)


def test_register_failure_does_not_add_price():
    t = AgentCostTracker()
    with pytest.raises(ValueError):
        t.register("bad", input_per_mtok=-1.0, output_per_mtok=1.0)
    assert t.has_price("bad") is False


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def test_record_returns_self():
    t = make_tracker()
    assert t.record(_MODEL, input_tokens=100, output_tokens=50) is t


def test_record_increments_call_count():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=50)
    assert t.call_count() == 1


def test_record_negative_input_raises():
    t = make_tracker()
    with pytest.raises(ValueError):
        t.record(_MODEL, input_tokens=-1, output_tokens=0)


def test_record_negative_output_raises():
    t = make_tracker()
    with pytest.raises(ValueError):
        t.record(_MODEL, input_tokens=0, output_tokens=-1)


def test_record_unknown_model_non_strict():
    t = AgentCostTracker(strict=False)
    t.record("mystery-model", input_tokens=100, output_tokens=50)
    assert t.call_count() == 1
    assert t.total_cost() == pytest.approx(0.0)


def test_record_unknown_model_strict():
    t = AgentCostTracker(strict=True)
    with pytest.raises(UnknownModelError):
        t.record("mystery-model", input_tokens=100, output_tokens=50)


def test_record_computes_cost():
    t = make_tracker()
    # 1_000_000 input at $2/MTok + 1_000_000 output at $8/MTok = $10
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
    assert t.total_cost() == pytest.approx(10.0)


def test_record_zero_tokens():
    t = make_tracker()
    t.record(_MODEL, input_tokens=0, output_tokens=0)
    assert t.call_count() == 1
    assert t.total_cost() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# token totals
# ---------------------------------------------------------------------------


def test_total_input_tokens():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500)
    t.record(_MODEL, input_tokens=2000, output_tokens=800)
    assert t.total_input_tokens() == 3000


def test_total_output_tokens():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500)
    t.record(_MODEL, input_tokens=2000, output_tokens=800)
    assert t.total_output_tokens() == 1300


def test_total_tokens():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500)
    assert t.total_tokens() == 1500


# ---------------------------------------------------------------------------
# cost_by_model / tokens_by_model / calls_by_model
# ---------------------------------------------------------------------------


def test_cost_by_model():
    t = make_tracker()
    t.register("m2", input_per_mtok=1.0, output_per_mtok=4.0)
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=0)
    t.record("m2", input_tokens=1_000_000, output_tokens=0)
    cbm = t.cost_by_model()
    assert cbm[_MODEL] == pytest.approx(2.0)
    assert cbm["m2"] == pytest.approx(1.0)


def test_cost_by_model_sorted():
    t = make_tracker()
    t.register("z-model", input_per_mtok=1, output_per_mtok=1)
    t.record(_MODEL, input_tokens=100, output_tokens=0)
    t.record("z-model", input_tokens=100, output_tokens=0)
    keys = list(t.cost_by_model().keys())
    assert keys == sorted(keys)


def test_tokens_by_model():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500)
    tbm = t.tokens_by_model()
    assert tbm[_MODEL]["input"] == 1000
    assert tbm[_MODEL]["output"] == 500
    assert tbm[_MODEL]["total"] == 1500


def test_calls_by_model():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=50)
    t.record(_MODEL, input_tokens=200, output_tokens=100)
    assert t.calls_by_model()[_MODEL] == 2


# ---------------------------------------------------------------------------
# cost_by_session / tokens_by_session / sessions
# ---------------------------------------------------------------------------


def test_cost_by_session():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=0, session="s1")
    t.record(_MODEL, input_tokens=500_000, output_tokens=0, session="s2")
    cbs = t.cost_by_session()
    assert cbs["s1"] == pytest.approx(2.0)
    assert cbs["s2"] == pytest.approx(1.0)


def test_cost_by_session_default_key():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=0)
    cbs = t.cost_by_session()
    assert "__default__" in cbs


def test_sessions_sorted():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=0, session="z")
    t.record(_MODEL, input_tokens=100, output_tokens=0, session="a")
    assert t.sessions() == ["a", "z"]


def test_sessions_excludes_none():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=0)  # session=None
    t.record(_MODEL, input_tokens=100, output_tokens=0, session="s1")
    assert t.sessions() == ["s1"]


def test_tokens_by_session():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500, session="s1")
    tbs = t.tokens_by_session()
    assert tbs["s1"]["input"] == 1000
    assert tbs["s1"]["total"] == 1500


def test_tokens_by_session_default_key():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1000, output_tokens=500)  # session=None
    tbs = t.tokens_by_session()
    assert tbs["__default__"]["input"] == 1000
    assert tbs["__default__"]["output"] == 500
    assert tbs["__default__"]["total"] == 1500


# ---------------------------------------------------------------------------
# records / reset
# ---------------------------------------------------------------------------


def test_records_structure():
    t = make_tracker()
    t.record(
        _MODEL, input_tokens=100, output_tokens=50, session="s1", metadata={"run": 1}
    )
    recs = t.records()
    assert len(recs) == 1
    r = recs[0]
    assert r["model"] == _MODEL
    assert r["input_tokens"] == 100
    assert r["output_tokens"] == 50
    assert r["session"] == "s1"
    assert r["metadata"] == {"run": 1}
    assert "cost_usd" in r


def test_records_is_copy():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=50)
    recs = t.records()
    recs[0]["model"] = "changed"
    assert t.records()[0]["model"] == _MODEL


def test_reset_clears_records():
    t = make_tracker()
    t.record(_MODEL, input_tokens=100, output_tokens=50)
    t.reset()
    assert t.call_count() == 0
    assert t.total_cost() == pytest.approx(0.0)


def test_reset_keeps_price_table():
    t = make_tracker()
    t.reset()
    assert t.has_price(_MODEL)


def test_reset_returns_self():
    t = make_tracker()
    assert t.reset() is t


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def test_summary_keys():
    s = make_tracker().summary()
    for key in (
        "calls",
        "total_input_tokens",
        "total_output_tokens",
        "total_tokens",
        "total_cost_usd",
        "by_model",
        "by_session",
    ):
        assert key in s


def test_summary_values():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
    s = t.summary()
    assert s["calls"] == 1
    assert s["total_cost_usd"] == pytest.approx(10.0)


def test_summary_breakdowns():
    t = make_tracker()
    t.record(_MODEL, input_tokens=1_000_000, output_tokens=0, session="s1")
    s = t.summary()
    assert s["by_model"][_MODEL] == pytest.approx(2.0)
    assert s["by_session"]["s1"] == pytest.approx(2.0)
    assert s["total_input_tokens"] == 1_000_000
    assert s["total_output_tokens"] == 0
