# Round 6 — ov_server integration feedback
**Against:** infergate v0.1.9
**Date:** 2026-05-14
**Integration goal this round:** Upgrade to 0.1.9, wire cache_stats into /health,
wire estimated_cost_usd into routing log. Found and fixed a profile tier routing bug
during live curl testing.

---

## What worked — do not regress

- **`Router.cache_stats()`** — confirmed working. Wired directly into `/health` as
  `embedding_cache: {hits, misses, size, capacity}`. Hit/miss tracking confirmed
  live: after 3 requests (2 unique queries, 1 repeat), stats read
  `{hits:1, misses:2, size:2, capacity:512}`. Exactly what was needed.

- **`RouteDecision.estimated_cost_usd`** — confirmed populated as `None` for all
  local models (no `cost_per_1k_tokens` set). Will populate automatically once
  cost data is added to `config.yaml` for OVH models. Included in routing log
  and in `last_routing_decision` dict at `/health`.

- **`RouteTrace.scope_source: Literal[...]`** — confirmed. Default `"global"` is
  semantically correct and shows up in debug trace logs.

---

## Bug found during testing — profile tier not reaching routing

**Severity:** High — profile switches had no effect on model selection.

`Router.decide()` reads `self._config.active_profile` which is frozen at
construction time from `config.yaml` (`active_profile: fast`). ov_server's
`/admin/profile` endpoint updates `app_state.active_profile` but never updates
infergate's internal config. Result: all requests used `model_preference: fastest`
regardless of the active profile. Under `laborious` profile (model_preference:
best), code requests still selected `qwen2.5-coder-14b-int4` (fast tier) instead
of `qwen3-coder-30b-a3b-int4-ov` (best tier).

**ov_server workaround (committed as `c5711b5`):**
After `decide()`, if the active profile's `model_preference` is not `"fastest"`,
call `reselect()` with `force_tier=model_preference`:
```python
_prof_pref = _cfg.get("profiles", {}).get(app_state.active_profile, {}).get("model_preference", "fastest")
if _prof_pref != "fastest":
    decision = app_state.ig_router.reselect(task_class=task_class, scope="local", force_tier=_prof_pref)
```
Strategy field now reads `"embedding+laborious"` etc. for traceability.

This works but is the wrong layer for this logic — the routing classification
(embedding, signal, directive) is correct; only the model selection step needs
to respect the active tier. The workaround duplicates part of `select_model()`.

---

## API friction — harder than it should be

### 1. No way to update active_profile at runtime without reconstructing Router

The only mechanism to override tier is `reselect()`, which requires the caller
to already know the task_class from a prior `decide()` call. There is no way to
tell `decide()` "use tier X for model selection" in a single call.

The workaround (decide → reselect) costs an extra synchronous call on every
non-fast-profile request, and duplicates the "apply tier preference" logic
that already lives inside `select_model()`.

---

## Proposed changes with rationale

- **P1** `Router.decide(request, *, trace=False, force_tier: str | None = None)`
  — add `force_tier` to `decide()`. When set, it overrides the profile's
  `model_preference` inside `select_model()`. This is the missing link between
  ov_server's runtime profile state and infergate's model selection.

  With this, ov_server's routing path becomes:
  ```python
  _prof_pref = _cfg["profiles"][app_state.active_profile].get("model_preference", "fastest")
  decision = await ig_router.decide(
      _ig_req,
      trace=debug_logging,
      force_tier=None if _prof_pref == "fastest" else _prof_pref,
  )
  ```
  One call. No reselect workaround. The `reselect()` path remains for cloud
  directive (scope change), which is a different concern.

- **P2** `Router.set_active_profile(name: str)` — alternative or complement to
  `force_tier` on `decide()`. Lets ov_server update infergate's profile at profile
  switch time, so `decide()` reads the right `model_preference` automatically.
  Simpler from ov_server's perspective, but requires Router to hold mutable
  runtime state (current profile name) beyond what the config YAML provides.
  Lower preference than P1 — P1 is more explicit and composable.

---

## Explicit non-requests

- **Do not add OVH HTTP proxy logic to infergate.**
- **Do not make `decide()` async when force_tier is set** — reselect is already
  synchronous and correct; force_tier should follow the same pattern.
- **Do not change the `reselect()` signature** — it is used correctly for
  cloud/scope changes and the current interface is clean.

---

## Upgrade delta

**From round 5 (v0.1.8 → v0.1.9):**

- P1 `Router.cache_stats()` — CONFIRMED. `{hits, misses, size, capacity}` all
  correct. Wired into `/health` as `embedding_cache`. Live-verified: 3 requests,
  2 unique, 1 hit, 1 miss correctly counted.

- P2 `RouteDecision.estimated_cost_usd` — CONFIRMED. `None` for all local models
  as expected. Will activate when `cost_per_1k_tokens` is added to OVH model
  entries in `config.yaml`.

- P2 `RouteTrace.scope_source` as `Literal` — CONFIRMED. Default `"global"` shows
  correctly in debug trace output.
