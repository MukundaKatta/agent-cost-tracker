"""Track USD cost per call and session based on token counts and a price table.

The tracker keeps a running total of input/output tokens and USD cost,
broken down per model and per optional session tag.

Example::

    from agent_cost_tracker import AgentCostTracker, ModelPrice

    tracker = AgentCostTracker()

    # Register custom model prices (USD per 1 000 000 tokens)
    tracker.register("my-model", input_per_mtok=3.00, output_per_mtok=15.00)

    # Record calls
    tracker.record("my-model", input_tokens=1024, output_tokens=512)
    tracker.record("my-model", input_tokens=2048, output_tokens=1024,
                   session="chat-1")

    print(tracker.total_cost())           # total USD spent
    print(tracker.cost_by_model())        # {model: usd}
    print(tracker.cost_by_session())      # {session: usd}
    print(tracker.summary())              # full breakdown dict
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# Built-in price table  (USD per 1 000 000 tokens = per-MTok)
# These are illustrative; override with register() for exact current pricing.
# ---------------------------------------------------------------------------

_BUILTIN_PRICES: dict[str, tuple[float, float]] = {
    # model: (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-3-5": (0.80, 4.00),
    "claude-opus-4": (15.00, 75.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-5-4": (2.50, 10.00),
    "o3": (10.00, 40.00),
    "o4-mini": (1.10, 4.40),
    "gemini-2-0-flash": (0.10, 0.40),
    "gemini-2-5-pro": (1.25, 10.00),
}


class UnknownModelError(KeyError):
    """Raised when a model name has no registered price entry."""


@dataclass
class ModelPrice:
    """Price entry for a single model.

    All prices are USD per 1 000 000 tokens (MTok).
    """

    model: str
    input_per_mtok: float
    output_per_mtok: float

    def __post_init__(self) -> None:
        if self.input_per_mtok < 0:
            raise ValueError(f"input_per_mtok must be >= 0, got {self.input_per_mtok}")
        if self.output_per_mtok < 0:
            raise ValueError(
                f"output_per_mtok must be >= 0, got {self.output_per_mtok}"
            )

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        """Return USD cost for *input_tokens* and *output_tokens*."""
        return (
            input_tokens * self.input_per_mtok / 1_000_000
            + output_tokens * self.output_per_mtok / 1_000_000
        )


@dataclass
class _CallRecord:
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    session: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class AgentCostTracker:
    """Track USD cost and token counts across LLM calls.

    Args:
        strict: When ``True``, :meth:`record` raises :class:`UnknownModelError`
                if the model is not in the price table.  When ``False``
                (default), unknown models are recorded with cost = 0.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self._strict = strict
        self._prices: dict[str, ModelPrice] = {}
        self._records: list[_CallRecord] = []
        # Load built-ins (can be overridden)
        for name, (inp, out) in _BUILTIN_PRICES.items():
            self._prices[name] = ModelPrice(
                model=name, input_per_mtok=inp, output_per_mtok=out
            )

    # ------------------------------------------------------------------
    # Price table
    # ------------------------------------------------------------------

    def register(
        self,
        model: str,
        *,
        input_per_mtok: float,
        output_per_mtok: float,
    ) -> AgentCostTracker:
        """Register or update a model price entry.

        Args:
            model:            Model name / identifier.
            input_per_mtok:   USD per 1 000 000 input tokens.
            output_per_mtok:  USD per 1 000 000 output tokens.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If either price is negative.
        """
        self._prices[model] = ModelPrice(
            model=model,
            input_per_mtok=input_per_mtok,
            output_per_mtok=output_per_mtok,
        )
        return self

    def price(self, model: str) -> ModelPrice:
        """Return the :class:`ModelPrice` entry for *model*.

        Raises:
            UnknownModelError: If *model* is not registered.
        """
        if model not in self._prices:
            raise UnknownModelError(model)
        return self._prices[model]

    def known_models(self) -> list[str]:
        """Return sorted list of registered model names."""
        return sorted(self._prices.keys())

    def has_price(self, model: str) -> bool:
        """Return ``True`` if *model* has a registered price."""
        return model in self._prices

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self,
        model: str,
        *,
        input_tokens: int,
        output_tokens: int,
        session: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentCostTracker:
        """Record a single LLM call.

        Args:
            model:         Model identifier (must be in the price table when
                           ``strict=True``).
            input_tokens:  Prompt/input token count.
            output_tokens: Completion/output token count.
            session:       Optional session tag for per-session aggregation.
            metadata:      Optional extra data attached to this record.

        Returns:
            ``self`` for chaining.

        Raises:
            UnknownModelError: In strict mode, if *model* is not registered.
            ValueError: If token counts are negative.
        """
        if input_tokens < 0:
            raise ValueError(f"input_tokens must be >= 0, got {input_tokens}")
        if output_tokens < 0:
            raise ValueError(f"output_tokens must be >= 0, got {output_tokens}")
        if model not in self._prices:
            if self._strict:
                raise UnknownModelError(model)
            cost = 0.0
        else:
            cost = self._prices[model].cost(input_tokens, output_tokens)
        self._records.append(
            _CallRecord(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                session=session,
                metadata=dict(metadata or {}),
            )
        )
        return self

    def reset(self) -> AgentCostTracker:
        """Clear all recorded calls (price table is kept)."""
        self._records.clear()
        return self

    # ------------------------------------------------------------------
    # Aggregates
    # ------------------------------------------------------------------

    def call_count(self) -> int:
        """Return the total number of recorded calls."""
        return len(self._records)

    def total_input_tokens(self) -> int:
        """Return total input tokens across all calls."""
        return sum(r.input_tokens for r in self._records)

    def total_output_tokens(self) -> int:
        """Return total output tokens across all calls."""
        return sum(r.output_tokens for r in self._records)

    def total_tokens(self) -> int:
        """Return total tokens (input + output) across all calls."""
        return self.total_input_tokens() + self.total_output_tokens()

    def total_cost(self) -> float:
        """Return total USD cost across all calls."""
        return sum(r.cost_usd for r in self._records)

    # ------------------------------------------------------------------
    # Per-model aggregates
    # ------------------------------------------------------------------

    def cost_by_model(self) -> dict[str, float]:
        """Return ``{model: total_usd}`` for each model seen."""
        result: dict[str, float] = {}
        for r in self._records:
            result[r.model] = result.get(r.model, 0.0) + r.cost_usd
        return dict(sorted(result.items()))

    def tokens_by_model(self) -> dict[str, dict[str, int]]:
        """Return ``{model: {input, output, total}}`` for each model."""
        result: dict[str, dict[str, int]] = {}
        for r in self._records:
            if r.model not in result:
                result[r.model] = {"input": 0, "output": 0, "total": 0}
            result[r.model]["input"] += r.input_tokens
            result[r.model]["output"] += r.output_tokens
            result[r.model]["total"] += r.input_tokens + r.output_tokens
        return dict(sorted(result.items()))

    def calls_by_model(self) -> dict[str, int]:
        """Return ``{model: call_count}`` for each model seen."""
        result: dict[str, int] = {}
        for r in self._records:
            result[r.model] = result.get(r.model, 0) + 1
        return dict(sorted(result.items()))

    # ------------------------------------------------------------------
    # Per-session aggregates
    # ------------------------------------------------------------------

    def cost_by_session(self) -> dict[str, float]:
        """Return ``{session: total_usd}`` for each named session.

        Calls with ``session=None`` are grouped under ``"__default__"``.
        """
        result: dict[str, float] = {}
        for r in self._records:
            key = r.session if r.session is not None else "__default__"
            result[key] = result.get(key, 0.0) + r.cost_usd
        return dict(sorted(result.items()))

    def tokens_by_session(self) -> dict[str, dict[str, int]]:
        """Return ``{session: {input, output, total}}`` for each session."""
        result: dict[str, dict[str, int]] = {}
        for r in self._records:
            key = r.session if r.session is not None else "__default__"
            if key not in result:
                result[key] = {"input": 0, "output": 0, "total": 0}
            result[key]["input"] += r.input_tokens
            result[key]["output"] += r.output_tokens
            result[key]["total"] += r.input_tokens + r.output_tokens
        return dict(sorted(result.items()))

    def sessions(self) -> list[str]:
        """Return sorted list of session names seen (excludes ``None``)."""
        return sorted({r.session for r in self._records if r.session is not None})

    # ------------------------------------------------------------------
    # Records
    # ------------------------------------------------------------------

    def records(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded calls as a list of dicts."""
        return [
            {
                "model": r.model,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
                "cost_usd": r.cost_usd,
                "session": r.session,
                "metadata": dict(r.metadata),
            }
            for r in self._records
        ]

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Return a high-level summary dict."""
        return {
            "calls": self.call_count(),
            "total_input_tokens": self.total_input_tokens(),
            "total_output_tokens": self.total_output_tokens(),
            "total_tokens": self.total_tokens(),
            "total_cost_usd": round(self.total_cost(), 8),
            "by_model": self.cost_by_model(),
            "by_session": self.cost_by_session(),
        }

    def __repr__(self) -> str:
        return (
            f"AgentCostTracker(calls={self.call_count()}, "
            f"total_cost_usd={self.total_cost():.6f})"
        )
