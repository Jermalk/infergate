# Response to Round 1 — infergate v0.1.1 → v0.1.2
**Date:** 2026-05-14
**Addresses:** `round_01_v0.1.1.md`
**Published:** infergate v0.1.2 on PyPI

---

## Done

- **Stale wheel (P0):** Rebuilt `infergate-0.1.2-py3-none-any.whl` from clean source.
  Verified wheel contains: `signal_only`, `tools_task_class`, `Modality`,
  `complexity_promote_fast_threshold`, `NoModelAvailable`, `force_tier`.
  The `AttributeError: 'TaskClassConfig' object has no attribute 'signal_only'`
  is resolved — wheel and source are now in sync.

- **Version bump:** 0.1.1 → 0.1.2. PyPI blocked re-upload of 0.1.1 filename;
  new version was the correct fix regardless.

## Skipped — with reason

- Nothing skipped. Round 1 had no API change requests beyond the stale wheel fix.

## Breaking changes

none — all new fields have defaults that preserve existing behaviour.

## What's new in 0.1.2 vs the broken 0.1.1 wheel

These fields were in source during round 1 but absent from the wheel:

| Addition | Where | Default |
|---|---|---|
| `TaskClassConfig.signal_only` | config.py | `False` |
| `RouterSettings.tools_task_class` | config.py | `"web_search"` |
| `RouterSettings.complexity_promote_fast_threshold` | config.py | `None` (disabled) |
| `ModelDescriptor.modality` | config.py | `"text"` |
| `NoModelAvailable` exception | types.py | — |
| `InferRequest.force_tier` | types.py | `None` |
| Dynamic directive regex at Router init | router.py | — |

## Upgrade notes

```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version   # expect: 0.1.2
```

No manual migration needed — all new fields default to previous behaviour.
