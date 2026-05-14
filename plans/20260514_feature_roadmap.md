# infergate — Feature Roadmap (post-v0.1.4)

**Date:** 2026-05-14
**infergate version at planning:** 0.1.4
**Source:** Design review session — identified gaps from real integration + forward analysis

---

## Order of implementation

| Step | Feature | Version target |
|---|---|---|
| 1+5 | Estimated tokens in `RouteDecision` + cost field on `ModelDescriptor` | 0.1.5 |
| 4   | Routing trace (`RouteTrace`) alongside `RouteDecision` | 0.1.6 |
| 2   | Embedding LRU cache (internal, no API change) | 0.1.7 |
| 3   | `Router.decide_batch()` | 0.1.8 |

Sequencing rationale: trace (4) before batch (3) so decide() is instrumented once and
batch inherits the infrastructure. Cache (2) before batch (3) so batch can rely on cache
for cross-request dedup instead of implementing its own.

---

## Step 1+5 — Estimated tokens + cost field (v0.1.5)

### 1. `RouteDecision.estimated_tokens: int`

`decide()` already computes `total_tokens` internally (line 137–139 of router.py) and
discards it. Surface it in the returned decision.

**Files:** `src/infergate/types.py`, `src/infergate/router.py`

**Change in types.py:**
```python
@dataclass
class RouteDecision:
    ...
    estimated_tokens: int = 0   # prompt token estimate (len(text) // 4)
```

**Change in router.py:** pass `estimated_tokens=total_tokens` in the RouteDecision constructor.

**Tests:** assert `decision.estimated_tokens > 0` for a non-trivial prompt.

---

### 5. `ModelDescriptor.cost_per_1k_tokens: float | None`

Config-only field. No routing logic changes in this step — purely metadata carried
through to `RouteDecision.model_id` / caller lookups. Future steps may use it for
cost-aware tier selection.

**Files:** `src/infergate/config.py`

**Change in config.py:**
```python
@dataclass
class ModelDescriptor:
    ...
    cost_per_1k_tokens: float | None = None  # USD; None = unknown/free
```

**Tests:** load a config with cost field set, assert it round-trips through RouterConfig.

---

## Step 4 — RouteTrace (v0.1.6)

Opt-in structured trace of the routing decision. Shows eliminated candidates with reasons.

**New type in types.py:**
```python
@dataclass
class EliminatedCandidate:
    model_id: str
    backend:  str
    reason:   str   # "scope", "ctx_limit", "modality", "no_backend", "unavailable"

@dataclass
class RouteTrace:
    eliminated:    list[EliminatedCandidate]
    scope_source:  str   # "class_override" | "cloud_directive" | "global"
    embedding_ms:  float | None   # wall time of embed() call; None if signal/keyword path
```

**API:** `Router.decide(request, trace=False) -> RouteDecision | tuple[RouteDecision, RouteTrace]`
Trace capture is opt-in — zero overhead when not requested.

**Files:** `src/infergate/types.py`, `src/infergate/router.py`, `src/infergate/selector.py`

selector.py must pass elimination reasons up to router.py when trace=True.

---

## Step 2 — Embedding LRU cache (v0.1.7)

Internal optimization. No API surface change.

Cache keyed on query text (exact string). LRU eviction. Default size: 512 entries.
Configurable via `RouterSettings.embedding_cache_size: int = 512` (0 = disabled).

**Files:** `src/infergate/router.py`, `src/infergate/config.py`

Cache lives on the `Router` instance. `load_embeddings()` does not populate it —
it fills on first `decide()` call per unique query.

If trace is active, report cache hit/miss in `RouteTrace`.

---

## Step 3 — decide_batch() (v0.1.8)

```python
async def decide_batch(
    self,
    requests: list[InferRequest],
    trace: bool = False,
) -> list[RouteDecision]:
```

Signal detection runs per-request (O(1), cheap). Uncached embedding queries are
collected, sent as a single `embed_batch()` call, results written to cache, then
used to complete per-request routing. Requests that hit cache or take the signal
path skip the batch embed entirely.

**Files:** `src/infergate/router.py`

**Tests:** assert that a batch of N identical queries triggers exactly one embed_batch()
call (mock the provider, count invocations).
