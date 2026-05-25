# agent-cost-tracker

Track USD cost per call and session based on token counts and a price table.

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

## License

MIT
