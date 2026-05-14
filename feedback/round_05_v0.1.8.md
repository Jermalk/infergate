# Round 5 — ov_server integration feedback
**Against:** infergate v0.1.8
**Date:** 2026-05-14
**Integration goal this round:** Upgrade from v0.1.4 → v0.1.8, wire estimated_tokens into
routing audit log, enable RouteTrace in debug mode, validate LRU cache safety.

---

## What worked — do not regress

- **LRU cache is correctly scoped.** `_EmbedCache` stores only
  `(task_class, confidence, embedding)`. `select_model()` always runs fresh
  and always calls `backend.loaded_model_ids()` on every `decide()` call —
  even on a cache hit. This means `prefer_loaded` is always accurate and the
  cache is safe across model load/unload events. No stale routing. Do not
  change this — moving model selection into the cache would break live VRAM
  awareness.

- **`text_content()` handles multimodal content.** VLM messages where `content`
  is a list of `{type, text, image_url}` dicts are handled transparently. ov_server
  passes all messages including VLM ones to `InferRequest` without pre-processing.
  No wrapper needed. Do not regress this.

- **`trace=True` zero-overhead guard is correct.** `eliminated` is allocated only
  when `trace=True` (`[] if trace else None`). Confirmed no overhead when disabled.
  The guard pattern is correct — keep it.

- **`ctx_limit` filtering in `select_model` works.** Models with `ctx_limit < estimated_tokens`
  are eliminated with reason `"ctx_limit"` and appear in `RouteTrace.eliminated`.
  This is the correct place for context-limit routing — ov_server no longer needs
  a parallel long-context bypass path.

---

## API friction — harder than it should be

### 1. No aggregate cache stats — flying blind on cache sizing

`_EmbedCache` exposes only `__len__`. There is no hit/miss counter. The only way
to measure cache effectiveness is `trace=True` on every request, which adds
allocation overhead we want to avoid in production.

ov_server exposes a `/metrics` endpoint. We want to include cache hit rate there.
Currently we can't — we'd have to add our own wrapping layer around `decide()` to
count calls. This is the wrong place for that state to live.

**What we need:** `Router.cache_stats() -> dict[str, int]`

```python
router.cache_stats()
# → {"hits": 1482, "misses": 347, "size": 512, "capacity": 512}
```

`hits` and `misses` reset on `Router` construction. `size` is current occupancy.
`capacity` is `embedding_cache_size` from config. We use this in `/metrics` and
to tune `embedding_cache_size` without enabling debug trace.

---

## Missing — needed but absent

### 1. `RouteTrace.scope_source` is plain `str` — not typed as `Literal`

The docstring lists three valid values: `"class_override"`, `"cloud_directive"`,
`"global"`. These are stable. The field type is `str = ""`, which gives no IDE
completion and no exhaustiveness check in match/case.

We compare this value in debug log formatting and in tests. A typo produces no
error at definition time.

**What we need:** change the field annotation:
```python
# before
scope_source: str = ""

# after
from typing import Literal
scope_source: Literal["class_override", "cloud_directive", "global"] = "global"
```

Non-breaking. The default changes from `""` to `"global"` which is the correct
semantic (when no override fires, scope comes from global config).

### 2. `RouteDecision.estimated_cost_usd` — not computed despite `cost_per_1k_tokens` existing

`ModelDescriptor.cost_per_1k_tokens: float | None` was added in v0.1.5.
`RouteDecision.estimated_tokens: int` was added in v0.1.5.

The selector already holds both the winning model's descriptor and the token
estimate. Computing `estimated_cost_usd` in `select_model()` costs one
multiplication — but every caller that wants cost tracking must redo this
computation externally, including a lookup into `config.yaml` to find the winning
model's descriptor again.

**What we need:**
```python
# RouteDecision
estimated_cost_usd: float | None = None
# Populated in select_model() when cost_per_1k_tokens is set on the winning model:
# estimated_cost_usd = winner.cost_per_1k_tokens * estimated_tokens / 1000
```

`None` when `cost_per_1k_tokens` is absent on the winning model (local models,
models without cost data). This keeps the field optional and non-breaking.

---

## Proposed changes with rationale

- **P1** `Router.cache_stats() -> dict[str, int]` — `hits`, `misses`, `size`,
  `capacity`. Internal counters on `_EmbedCache`. We expose `/metrics`; without
  this we cannot report cache effectiveness without per-request trace overhead.

- **P2** `RouteDecision.estimated_cost_usd: float | None` — computed once in
  `select_model()` from the winner's `cost_per_1k_tokens * estimated_tokens / 1000`.
  `None` when cost data absent. Saves every caller from re-deriving it. Once we
  add OVH model costs to `config.yaml` this will populate automatically.

- **P2** `RouteTrace.scope_source: Literal["class_override", "cloud_directive", "global"]`
  with default `"global"` — non-breaking. The three values are stable and
  document-worthy. Default `""` is currently a sentinel for "not set"; `"global"`
  is the correct semantic.

---

## Explicit non-requests

- **Do not add OVH HTTP proxy logic to infergate.** The proxy dispatch
  (`_proxy_chat`, `api_key_env`, endpoint URL resolution) is ov_server-specific.
- **Do not change `EliminationReason` from `Literal` to `Enum`.** It works as
  a plain string in logging (`f"{c.reason}"`) and switching to Enum would require
  `.value` everywhere — a breaking change for no practical gain.
- **Do not move model selection into the embedding cache.** The current design
  (cache embeddings only, run `select_model` fresh each call) is correct. It
  preserves live `prefer_loaded` accuracy. Do not trade that for fewer allocations.
- **Do not add async to `reselect()`.** The synchronous design is intentional
  and correct given that `available_models()` is a dict lookup with no I/O.

---

## Upgrade delta

**From round 4 (v0.1.4 → v0.1.8) — proactive release:**

- v0.1.5 `RouteDecision.estimated_tokens` — CONFIRMED. Logged on every request.
  One line added to routing audit log: `tokens≈N`. Exactly what was described.

- v0.1.5 `ModelDescriptor.cost_per_1k_tokens` — noted, not yet used. OVH pricing
  data not yet added to `config.yaml`. Will wire once P2 `estimated_cost_usd` lands.

- v0.1.6 `RouteTrace` / `decide(trace=True)` — CONFIRMED. Wired behind
  `app_state.debug_logging`. Log line format:
  `[infergate:trace] scope_source=global embed_ms=12.3 cache=miss eliminated=[model-x(ctx_limit)]`
  Zero overhead when debug is off — confirmed by source inspection.

- v0.1.7 Embedding LRU cache — CONFIRMED active. Safe across model lifecycle events
  (see "What worked" section). Cannot measure hit rate yet — P1 `cache_stats()` needed.

- v0.1.8 `Router.decide_batch()` — noted, not applicable to ov_server's
  single-request async workload. No integration planned.
