# Response to Round 3 — infergate v0.1.3 → v0.1.4
**Date:** 2026-05-14
**Addresses:** `round_03_v0.1.3.md`
**Published:** infergate v0.1.4 on PyPI

---

## Done

- **P1 — `Router.reselect(task_class, scope, force_tier, complexity, estimated_tokens) → RouteDecision`**
  Synchronous public method on `Router`. Accepts a `task_class` already determined by `decide()`,
  re-runs `select_model()` with caller-supplied `scope` and `force_tier` overrides. Returns a new
  `RouteDecision` with `strategy=RouteStrategy.RESELECT` and `confidence=1.0`. No signal detection,
  no embedding encoding. Accepted scope strings: `"local"`, `"remote"`, `"local+remote"`.
  Unblocks deletion of `router._select_model()` and the parallel `config.json` task-class catalogue.

- **P2 — `RouteDecision.task_directive: str | None = None`**
  Added field to `RouteDecision`. `Router.decide()` sets it to the matched directive name
  (e.g. `"code"`) when a `#task-class` hashtag fired, `None` otherwise. No behaviour change.
  Callers can now read the detected directive from the decision rather than running a second
  detection pass with a frozen regex.

- **`RouteStrategy.RESELECT = "reselect"`** added to the enum for introspection by callers.

- 8 new tests covering both changes (87/87 pass).

## Skipped — with reason

- `has_cloud_directive()` top-level re-export: explicitly a non-request in the round file.
  `from infergate.signals import has_cloud_directive` is already usable.

## Breaking changes

None for callers already using positional construction of `RouteDecision`. The new fields
(`task_directive`) are keyword-only with defaults, so existing code is unaffected.

Third-party `RouteDecision` constructors that pass all fields positionally will break —
but no such callers are known (this is a return-only type in normal usage).

## Upgrade notes

```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version
```

After upgrade, the cloud-directive path in ov_server can be refactored:

```python
# Before (ov_server router.py):
if _cloud_directive:
    cplx = router.complexity_score(req)
    est_tokens = sum(len(_text_content(m)) for m in req.messages) // 4
    model_entry = router._select_model(
        task_class, active_profile_cfg, cplx, est_tokens,
        scope_override="local+ovh", pref_override="best",
    )
    model_id = model_entry["id"]

# After:
if _cloud_directive:
    decision = router.reselect(
        task_class=decision.task_class,
        scope="local+remote",
        force_tier="best",
    )
```

`decision.task_directive` is now set on the initial `decide()` result — no need for a
separate `router.task_class_directive(req.messages)` call.
