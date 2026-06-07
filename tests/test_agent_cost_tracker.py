"""Tests for agent_cost_tracker.

These tests use only the Python standard library (``unittest``) so they can be
run with zero third-party dependencies::

    python3 -m unittest discover -s tests
"""

from __future__ import annotations

import os
import sys
import unittest

# Support a bare ``python3 -m unittest discover -s tests`` run from the repo
# root: this project uses a ``src/`` layout, so add ``src`` to the import path
# when the package has not been installed. An installed package is found first
# on the path, so this only takes effect for an uninstalled checkout.
_SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if os.path.isdir(_SRC) and _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from agent_cost_tracker import (  # noqa: E402  (import after sys.path setup)
    AgentCostTracker,
    ModelPrice,
    UnknownModelError,
)

# We use a simple test model for deterministic cost math.
_MODEL = "test-model"
_INP_RATE = 2.0  # $2 per MTok input
_OUT_RATE = 8.0  # $8 per MTok output


def make_tracker() -> AgentCostTracker:
    t = AgentCostTracker()
    t.register(_MODEL, input_per_mtok=_INP_RATE, output_per_mtok=_OUT_RATE)
    return t


class ModelPriceTests(unittest.TestCase):
    def test_model_price_cost_zero(self):
        mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
        self.assertAlmostEqual(mp.cost(0, 0), 0.0)

    def test_model_price_cost_input_only(self):
        mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
        # 1_000_000 input tokens at $2/MTok = $2
        self.assertAlmostEqual(mp.cost(1_000_000, 0), 2.0)

    def test_model_price_cost_output_only(self):
        mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
        self.assertAlmostEqual(mp.cost(0, 1_000_000), 8.0)

    def test_model_price_cost_combined(self):
        mp = ModelPrice("m", input_per_mtok=2.0, output_per_mtok=8.0)
        # 500k input + 250k output
        expected = 500_000 * 2.0 / 1_000_000 + 250_000 * 8.0 / 1_000_000
        self.assertAlmostEqual(mp.cost(500_000, 250_000), expected)


class ConstructorReprTests(unittest.TestCase):
    def test_repr(self):
        t = AgentCostTracker()
        self.assertIn("calls=0", repr(t))
        self.assertIn("total_cost_usd=0.000000", repr(t))

    def test_initial_empty(self):
        t = AgentCostTracker()
        self.assertEqual(t.call_count(), 0)
        self.assertAlmostEqual(t.total_cost(), 0.0)
        self.assertEqual(t.total_tokens(), 0)


class PriceTableTests(unittest.TestCase):
    def test_register_returns_self(self):
        t = AgentCostTracker()
        self.assertIs(t.register("m", input_per_mtok=1.0, output_per_mtok=2.0), t)

    def test_register_overrides(self):
        t = AgentCostTracker()
        t.register("m", input_per_mtok=1.0, output_per_mtok=2.0)
        t.register("m", input_per_mtok=5.0, output_per_mtok=10.0)
        self.assertAlmostEqual(t.price("m").input_per_mtok, 5.0)

    def test_register_negative_input_rate_raises(self):
        t = AgentCostTracker()
        with self.assertRaises(ValueError):
            t.register("m", input_per_mtok=-1.0, output_per_mtok=2.0)

    def test_register_negative_output_rate_raises(self):
        t = AgentCostTracker()
        with self.assertRaises(ValueError):
            t.register("m", input_per_mtok=1.0, output_per_mtok=-2.0)

    def test_has_price_true(self):
        t = make_tracker()
        self.assertTrue(t.has_price(_MODEL))

    def test_has_price_false(self):
        self.assertFalse(AgentCostTracker().has_price("nope"))

    def test_known_models_sorted(self):
        t = AgentCostTracker()
        t.register("z", input_per_mtok=1, output_per_mtok=1)
        t.register("a", input_per_mtok=1, output_per_mtok=1)
        models = t.known_models()
        self.assertLess(models.index("a"), models.index("z"))

    def test_price_unknown_raises(self):
        with self.assertRaises(UnknownModelError):
            AgentCostTracker().price("nope")

    def test_builtin_prices_loaded(self):
        t = AgentCostTracker()
        self.assertTrue(t.has_price("claude-sonnet-4-6"))
        self.assertTrue(t.has_price("gpt-4o"))


class RecordTests(unittest.TestCase):
    def test_record_returns_self(self):
        t = make_tracker()
        self.assertIs(t.record(_MODEL, input_tokens=100, output_tokens=50), t)

    def test_record_increments_call_count(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=50)
        self.assertEqual(t.call_count(), 1)

    def test_record_negative_input_raises(self):
        t = make_tracker()
        with self.assertRaises(ValueError):
            t.record(_MODEL, input_tokens=-1, output_tokens=0)

    def test_record_negative_output_raises(self):
        t = make_tracker()
        with self.assertRaises(ValueError):
            t.record(_MODEL, input_tokens=0, output_tokens=-1)

    def test_record_unknown_model_non_strict(self):
        t = AgentCostTracker(strict=False)
        t.record("mystery-model", input_tokens=100, output_tokens=50)
        self.assertEqual(t.call_count(), 1)
        self.assertAlmostEqual(t.total_cost(), 0.0)

    def test_record_unknown_model_strict(self):
        t = AgentCostTracker(strict=True)
        with self.assertRaises(UnknownModelError):
            t.record("mystery-model", input_tokens=100, output_tokens=50)

    def test_record_computes_cost(self):
        t = make_tracker()
        # 1_000_000 input at $2/MTok + 1_000_000 output at $8/MTok = $10
        t.record(_MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(t.total_cost(), 10.0)

    def test_record_zero_tokens(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=0, output_tokens=0)
        self.assertEqual(t.call_count(), 1)
        self.assertAlmostEqual(t.total_cost(), 0.0)

    def test_record_metadata_is_copied(self):
        t = make_tracker()
        meta = {"run": 1}
        t.record(_MODEL, input_tokens=10, output_tokens=5, metadata=meta)
        meta["run"] = 999
        self.assertEqual(t.records()[0]["metadata"], {"run": 1})


class TokenTotalsTests(unittest.TestCase):
    def test_total_input_tokens(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1000, output_tokens=500)
        t.record(_MODEL, input_tokens=2000, output_tokens=800)
        self.assertEqual(t.total_input_tokens(), 3000)

    def test_total_output_tokens(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1000, output_tokens=500)
        t.record(_MODEL, input_tokens=2000, output_tokens=800)
        self.assertEqual(t.total_output_tokens(), 1300)

    def test_total_tokens(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1000, output_tokens=500)
        self.assertEqual(t.total_tokens(), 1500)


class PerModelAggregateTests(unittest.TestCase):
    def test_cost_by_model(self):
        t = make_tracker()
        t.register("m2", input_per_mtok=1.0, output_per_mtok=4.0)
        t.record(_MODEL, input_tokens=1_000_000, output_tokens=0)
        t.record("m2", input_tokens=1_000_000, output_tokens=0)
        cbm = t.cost_by_model()
        self.assertAlmostEqual(cbm[_MODEL], 2.0)
        self.assertAlmostEqual(cbm["m2"], 1.0)

    def test_cost_by_model_sorted(self):
        t = make_tracker()
        t.register("z-model", input_per_mtok=1, output_per_mtok=1)
        t.record(_MODEL, input_tokens=100, output_tokens=0)
        t.record("z-model", input_tokens=100, output_tokens=0)
        keys = list(t.cost_by_model().keys())
        self.assertEqual(keys, sorted(keys))

    def test_tokens_by_model(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1000, output_tokens=500)
        tbm = t.tokens_by_model()
        self.assertEqual(tbm[_MODEL]["input"], 1000)
        self.assertEqual(tbm[_MODEL]["output"], 500)
        self.assertEqual(tbm[_MODEL]["total"], 1500)

    def test_calls_by_model(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=50)
        t.record(_MODEL, input_tokens=200, output_tokens=100)
        self.assertEqual(t.calls_by_model()[_MODEL], 2)


class PerSessionAggregateTests(unittest.TestCase):
    def test_cost_by_session(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1_000_000, output_tokens=0, session="s1")
        t.record(_MODEL, input_tokens=500_000, output_tokens=0, session="s2")
        cbs = t.cost_by_session()
        self.assertAlmostEqual(cbs["s1"], 2.0)
        self.assertAlmostEqual(cbs["s2"], 1.0)

    def test_cost_by_session_default_key(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1_000_000, output_tokens=0)
        cbs = t.cost_by_session()
        self.assertIn("__default__", cbs)

    def test_sessions_sorted(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=0, session="z")
        t.record(_MODEL, input_tokens=100, output_tokens=0, session="a")
        self.assertEqual(t.sessions(), ["a", "z"])

    def test_sessions_excludes_none(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=0)  # session=None
        t.record(_MODEL, input_tokens=100, output_tokens=0, session="s1")
        self.assertEqual(t.sessions(), ["s1"])

    def test_tokens_by_session(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1000, output_tokens=500, session="s1")
        tbs = t.tokens_by_session()
        self.assertEqual(tbs["s1"]["input"], 1000)
        self.assertEqual(tbs["s1"]["total"], 1500)


class RecordsResetTests(unittest.TestCase):
    def test_records_structure(self):
        t = make_tracker()
        t.record(
            _MODEL,
            input_tokens=100,
            output_tokens=50,
            session="s1",
            metadata={"run": 1},
        )
        recs = t.records()
        self.assertEqual(len(recs), 1)
        r = recs[0]
        self.assertEqual(r["model"], _MODEL)
        self.assertEqual(r["input_tokens"], 100)
        self.assertEqual(r["output_tokens"], 50)
        self.assertEqual(r["session"], "s1")
        self.assertEqual(r["metadata"], {"run": 1})
        self.assertIn("cost_usd", r)

    def test_records_is_copy(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=50)
        recs = t.records()
        recs[0]["model"] = "changed"
        self.assertEqual(t.records()[0]["model"], _MODEL)

    def test_records_metadata_is_copy(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=50, metadata={"k": "v"})
        recs = t.records()
        recs[0]["metadata"]["k"] = "tampered"
        self.assertEqual(t.records()[0]["metadata"], {"k": "v"})

    def test_reset_clears_records(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=100, output_tokens=50)
        t.reset()
        self.assertEqual(t.call_count(), 0)
        self.assertAlmostEqual(t.total_cost(), 0.0)

    def test_reset_keeps_price_table(self):
        t = make_tracker()
        t.reset()
        self.assertTrue(t.has_price(_MODEL))

    def test_reset_returns_self(self):
        t = make_tracker()
        self.assertIs(t.reset(), t)


class SummaryTests(unittest.TestCase):
    def test_summary_keys(self):
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
            self.assertIn(key, s)

    def test_summary_values(self):
        t = make_tracker()
        t.record(_MODEL, input_tokens=1_000_000, output_tokens=1_000_000)
        s = t.summary()
        self.assertEqual(s["calls"], 1)
        self.assertAlmostEqual(s["total_cost_usd"], 10.0)


class ChainingTests(unittest.TestCase):
    def test_record_can_be_chained(self):
        t = make_tracker()
        result = (
            t.record(_MODEL, input_tokens=100, output_tokens=50)
            .record(_MODEL, input_tokens=200, output_tokens=100)
            .record(_MODEL, input_tokens=300, output_tokens=150)
        )
        self.assertIs(result, t)
        self.assertEqual(t.call_count(), 3)


if __name__ == "__main__":
    unittest.main()
