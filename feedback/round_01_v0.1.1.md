# Round 1 — ov_server integration feedback
**Against:** infergate v0.1.1
**Date:** 2026-05-14
**Integration goal this round:** Install infergate, parse config.yaml, verify RouterConfig

---

## What worked — do not regress
- `RouterConfig.from_dict()` parsed all 5 task_classes correctly from YAML
- `signal_only: true` on vision and web_search classes was picked up cleanly
- `ctx_limit` / `max_context_tokens` backward-compat shim worked without any config change
- `modality: vision` on VLM models parsed and stored correctly
- `provider` → `backend` alias in `from_dict()` worked — no manual migration needed
- Profile names (fast/precise/laborious) survived round-trip unchanged

## API friction — harder than it should be
- Nothing significant at this stage. Config parsing was straightforward.

## Missing — needed but absent
- Nothing discovered at config-parsing stage.

## Proposed changes with rationale
- None at this stage.

## Explicit non-requests
- Do not add ov_server-specific fields (device, kv_cache_size_gb, thinking, max_new_tokens)
  to RouterConfig — those are inference parameters that stay in config.json.

## Upgrade delta
N/A (round 1)

---

## Bug found: published wheel does not match source — BLOCKS integration

**Severity: P0**

The `infergate-0.1.1-py3-none-any.whl` in `dist/` is a stale build predating all 8 gap
fixes confirmed in the source code review (2026-05-14). Installing from wheel fails
immediately:

```
AttributeError: 'TaskClassConfig' object has no attribute 'signal_only'
```

ov_server integration is paused on an editable source install (`pip install -e .`).
We will not continue building adapters against a dev install — the point of this track
is to validate infergate as a proper dependency, not to co-develop against a checkout.

**Expected next version from infergate session:**
- All 8 gaps from `plans/20260514_infergate_gaps.md` present in the built artifact
- Clean wheel built from current source (`python -m build`)
- Published to PyPI (or provided as a fresh wheel in `dist/`)
- Version bumped (suggest 0.1.2) so `pip install --upgrade infergate` picks it up

**When SIGNAL.md flips to RELEASE READY, ov_server session will:**
```bash
source /home/jerzy/ov_env/bin/activate
pip install --upgrade infergate
pip show infergate | grep Version   # confirm new version
```
Then continue with `ov_backend.py`.
