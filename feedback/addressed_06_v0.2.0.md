# Response to Round 6 — infergate v0.1.9 → v0.2.0
**Date:** 2026-05-14
**Addresses:** `round_06_v0.1.9.md`
**Published:** infergate v0.2.0 on PyPI

---

## Done

- **P1 `Router.decide(request, *, trace=False, force_tier: str | None = None)`** — implemented.
  When `force_tier` is set on the `decide()` call, it overrides the active profile's
  `model_preference` inside `select_model()`. Takes precedence over `request.force_tier`.
  When `force_tier=None` (default), `request.force_tier` is used as before — no behaviour
  change for existing callers.

  Same parameter added to `decide_batch()` for consistency, applying to all requests in
  the batch.

  ov_server integration path (replaces the decide→reselect workaround):
  ```python
  _prof_pref = _cfg["profiles"][app_state.active_profile].get("model_preference", "fastest")
  decision = await ig_router.decide(
      _ig_req,
      trace=debug_logging,
      force_tier=None if _prof_pref == "fastest" else _prof_pref,
  )
  ```

## Skipped — with reason

- **P2 `Router.set_active_profile(name: str)`** — skipped per round 6 explicit preference.
  P1 is more composable; mutable runtime state on Router adds complexity for no gain when
  the caller already knows the active profile.

## Breaking changes

none — `force_tier` is a keyword-only argument with default `None`. All existing call sites
continue to work without modification.

## Upgrade notes
```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version
```
Remove the `decide → reselect` workaround (commit `c5711b5`) once on v0.2.0.
