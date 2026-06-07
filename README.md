# agent-cost-tracker

Track USD cost per call and session based on token counts and a price table.

A tiny, dependency-free helper for accounting LLM/agent spend. Feed it the input
and output token counts of each model call and it keeps running totals of cost
and tokens, broken down per model and per (optional) session tag.

```python
from agent_cost_tracker import AgentCostTracker

tracker = AgentCostTracker()

# Built-in prices for common models (claude-sonnet-4-6, gpt-4o, etc.)
# Override or add your own:
tracker.register("my-model", input_per_mtok=3.00, output_per_mtok=15.00)

tracker.record("claude-sonnet-4-6", input_tokens=1024, output_tokens=512)
tracker.record("claude-sonnet-4-6", input_tokens=2048, output_tokens=1024,
               session="chat-1")

print(tracker.total_cost())           # total USD
print(tracker.cost_by_model())        # {model: usd}
print(tracker.cost_by_session())      # {session: usd}
print(tracker.summary())
```

## Install

```bash
pip install agent-cost-tracker
```

## Features

- Built-in price table for Claude, GPT-4o, Gemini, and o-series models
- `register(model, input_per_mtok, output_per_mtok)` — add or override prices
- `record(model, input_tokens, output_tokens, session, metadata)` — log a call
- `strict=True` raises `UnknownModelError` for unregistered models
- Input validation: negative token counts (`record`) and negative prices
  (`register`) raise `ValueError`
- `total_cost()`, `total_input_tokens()`, `total_output_tokens()`, `total_tokens()`
- `cost_by_model()`, `tokens_by_model()`, `calls_by_model()` — per-model breakdown
- `cost_by_session()`, `tokens_by_session()`, `sessions()` — per-session breakdown
- `records()` — full list of individual call records
- `summary()` — single-dict overview
- `reset()` — clear records without losing the price table
- Zero dependencies

## API

```python
t = AgentCostTracker(*, strict=False)

t.register(model, *, input_per_mtok, output_per_mtok)  -> self
t.record(model, *, input_tokens, output_tokens,
         session=None, metadata=None)                   -> self
t.reset()                                               -> self

t.total_cost()               -> float
t.total_input_tokens()       -> int
t.total_output_tokens()      -> int
t.total_tokens()             -> int
t.call_count()               -> int

t.cost_by_model()            -> dict[str, float]
t.tokens_by_model()          -> dict[str, dict]
t.calls_by_model()           -> dict[str, int]

t.cost_by_session()          -> dict[str, float]
t.tokens_by_session()        -> dict[str, dict]
t.sessions()                 -> list[str]

t.price(model)               -> ModelPrice
t.has_price(model)           -> bool
t.known_models()             -> list[str]

t.records()                  -> list[dict]
t.summary()                  -> dict
```

### Notes

- All prices in the built-in table and in `register()` are **USD per
  1,000,000 tokens** (per-MTok). The shipped numbers are illustrative; call
  `register()` to set the exact current pricing for your provider.
- Calls recorded without a `session` are grouped under the `"__default__"`
  key in `cost_by_session()` / `tokens_by_session()`, but are excluded from
  `sessions()`.
- The package ships type hints and a `py.typed` marker (PEP 561), so type
  checkers will pick up its annotations.

## Development

This project has **zero runtime and test dependencies**. The test suite uses
the standard-library `unittest` runner:

```bash
python -m unittest discover -s tests -v
```

(from the repo root; an editable install with `pip install -e .` puts the
package on the path, or set `PYTHONPATH=src`).

Optional linting/formatting (requires the `dev` extra):

```bash
pip install -e ".[dev]"
ruff check src tests
ruff format --check src tests
```

## License

MIT
