# Response to Round 5 — infergate v0.1.8 → v0.1.9
**Date:** 2026-05-14
**Addresses:** `round_05_v0.1.8.md`
**Published:** infergate v0.1.9 on PyPI

---

## Done

- **P1 `Router.cache_stats() -> dict[str, int]`** — implemented. `_EmbedCache` tracks `_hits`
  and `_misses` as integer counters, incremented in `get()`. `Router.cache_stats()` returns
  `{"hits", "misses", "size", "capacity"}`. Counters reset on `Router` construction.
  `capacity` reflects `embedding_cache_size` from config. Use directly in `/metrics` without
  enabling per-request trace overhead.

- **P2 `RouteDecision.estimated_cost_usd: float | None`** — implemented. Computed in
  `select_model()` as `winner.cost_per_1k_tokens * estimated_tokens / 1000`. Returns `None`
  when `cost_per_1k_tokens` is absent on the winning model OR when `estimated_tokens == 0`.
  Propagated through `decide()`, `decide_batch()`, and `reselect()`.

- **P2 `RouteTrace.scope_source: Literal["class_override", "cloud_directive", "global"]`**
  — implemented. Default changed from `""` to `"global"` (correct semantic: when no override
  fires, scope comes from global config). Non-breaking — all three values were already the
  only ones assigned in the codebase.

## Skipped — with reason

Nothing skipped.

## Breaking changes

none — all three changes are additive. `scope_source` default change (`""` → `"global"`) is
a semantic fix; any code checking `scope_source == ""` would have been a bug, not a feature.

## Upgrade notes
```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version
```
