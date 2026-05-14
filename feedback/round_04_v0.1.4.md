# Round 4 — ov_server integration feedback
**Against:** infergate v0.1.4
**Date:** 2026-05-14
**Integration goal this round:** Wire Router.reselect() for cloud directive, read
task_directive from RouteDecision, register OVHBackend alongside OVServerBackend.
Verify router.py is reduced to state + embedding loader only.

---

## What worked — do not regress

- `Router.reselect(task_class, scope, force_tier)` signature is exactly right.
  Synchronous, returns `RouteDecision`, no side-effects. Used in two places in
  ov_server: cloud directive path and profile-switch coexistence prediction.
- `RouteDecision.task_directive` — populated by `decide()`, None when no directive
  fires. Eliminated the second router.task_class_directive() pass entirely.
  Now covers all task classes in config.yaml, not just the three in the old
  hardcoded regex.
- `RouteStrategy.RESELECT` — readable in logs. `decision.strategy.value` works as
  expected for both RESELECT and existing values.
- OVHBackend registered alongside OVServerBackend: infergate's selector now
  considers OVH models when scope="local+remote". Silent skip when OVH catalogue
  cache is empty (OVH not configured) — correct fallback to best local model.
- `decision.backend == "ov_server"` vs `"ovh"` gives a clean boolean for the proxy
  branch check, replacing the fragile `model_entry["provider"] != "loc"` string.

## What was deleted from ov_server after this integration

The following were removed — do not reintroduce:

| Deleted | Replaced by |
|---|---|
| `router._select_model()` | `_ig_router.reselect()` |
| `router.complexity_score()` | (not needed — reselect handles it internally) |
| `router._has_cloud_directive()` | `infergate.signals.has_cloud_directive()` |
| `router.task_class_directive()` | `decision.task_directive` |
| `router._fastest_from/balanced_from/best_from()` | (internal to infergate) |
| `router._routing_prompt_cache` | (dead code — routing prompts no longer used) |
| 19 unit tests for the above | (behaviour now covered by infergate's own tests) |

`router.py` now contains only:
- `_last_routing_decision: dict | None` — read by /health
- `_load_embedding_centroids()` — ensures emb_model is in globals before infergate starts

---

## API friction — harder than it should be

### 1. `reselect()` is synchronous but calls `select_model()` which calls `backend.available_models()`

`OVHBackend.available_models()` reads from `catalogue._catalogue_cache`. The call
is fast (just a dict lookup), but it happens synchronously inside `reselect()`.
If the catalogue TTL check or network fetch were ever added here, it would block
the event loop. For now the current pattern is safe — noting it for the record.

### 2. No way to pass profile name to `reselect()`

`reselect()` accepts `force_tier` but not a profile name. In the profile-switch
coexistence check, ov_server maps `prof["model_preference"]` → `force_tier`:

```python
_pref = prof.get("model_preference", "balanced")
_target = _ig_router.reselect("general", scope="local", force_tier=_pref)
```

This works because ov_server's `model_preference` values ("fastest", "balanced",
"best") happen to match infergate's tier vocabulary exactly. If ov_server ever
adds a profile with a different preference vocabulary, the mapping would break.
Low risk for now — documenting as a latent coupling.

---

## Missing — nothing blocking

No blocking gaps. router.py is now effectively empty of logic. The only remaining
ov_server-specific routing code is the OVH proxy dispatch (using `config.json`
routing.backends spec) — this is correctly ov_server-specific and should not
move into infergate.

---

## Proposed changes with rationale

None — round 4 is a confirmation round. P1 and P2 from round 3 were implemented
exactly as requested. No new proposals.

---

## Explicit non-requests

- **Do not add OVH HTTP proxy logic to infergate.** The proxy dispatch
  (`_proxy_chat`, `api_key_env`, endpoint URL resolution) is ov_server-specific.
- **Do not add `catalogue._catalogue_cache` access to infergate internals.**
  OVHBackend reads it in ov_server's adapter code, not in the library.

---

## Upgrade delta

**From round 3 (v0.1.3 → v0.1.4):**
- P1 `Router.reselect()` — CONFIRMED working. Cloud directive and profile-switch
  coexistence both use it. router._select_model() deleted.
- P2 `RouteDecision.task_directive` — CONFIRMED. `decision.task_directive` is
  "code" when `#code` directive fires; None otherwise. task_class_directive()
  deleted from router.py.
- OVHBackend registered: `decision.backend` correctly distinguishes "ov_server"
  vs "ovh" without string-comparing provider fields.

router.py is now 23 lines. Integration complete through round 4.
