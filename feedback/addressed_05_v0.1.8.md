# Proactive release — infergate v0.1.5 → v0.1.8
**Date:** 2026-05-14
**Not in response to a round file** — internal feature roadmap shipped independently.
**Published:** infergate v0.1.8 on PyPI

---

## What's new (v0.1.5 – v0.1.8)

All changes are additive. No breaking changes. Existing ov_server adapter code requires
no modification to upgrade.

### v0.1.5 — RouteDecision.estimated_tokens + ModelDescriptor.cost_per_1k_tokens

- `RouteDecision.estimated_tokens: int` — prompt token estimate (sum of message text
  lengths // 4) is now surfaced in every decision. Previously computed internally and
  discarded. Useful for audit logs and cost tracking without re-computing in the caller.

- `ModelDescriptor.cost_per_1k_tokens: float | None` — optional USD cost field on each
  model descriptor in `config.yaml`. No routing logic change — purely metadata carried
  through to the decision for callers that want cost-aware logging or dashboards.
  Example config:
  ```yaml
  models:
    - id: meta-llama/Meta-Llama-3.1-70B-Instruct
      backend: ovh
      tier: best
      cost_per_1k_tokens: 0.00072
  ```

### v0.1.6 — RouteTrace (opt-in routing explainability)

`Router.decide(request, trace=True)` populates `RouteDecision.trace` with a `RouteTrace`:

```python
@dataclass
class RouteTrace:
    eliminated:   list[EliminatedCandidate]  # every model considered and why it was dropped
    scope_source: str           # "class_override" | "cloud_directive" | "global"
    embedding_ms: float | None  # wall time of embed() call; None on signal/keyword path
    cache_hit:    bool | None   # True/False on embedding path; None when path not taken
```

`EliminatedCandidate.reason` is one of: `"no_backend"`, `"scope"`, `"unavailable"`,
`"ctx_limit"`, `"modality"`.

Zero overhead when `trace=False` (the default). Useful for debugging routing
misconfiguration and for the `X-Routing-Decision` response header.

### v0.1.7 — Embedding LRU cache

Repeated `decide()` calls with identical prompts no longer re-encode. The cache lives
on the `Router` instance, keyed by query text, default size 512 entries.

Configurable in `config.yaml`:
```yaml
router:
  embedding_cache_size: 512   # 0 = disabled
```

`RouteTrace.cache_hit` reports `True`/`False`/`None` when trace is active.

### v0.1.8 — Router.decide_batch()

```python
decisions = await router.decide_batch(requests, trace=False)
```

Routes a list of `InferRequest` objects sharing a single `embed_batch()` call for
uncached queries. Signal/keyword requests and cache hits skip the batch embed entirely.
Useful for batch inference pipelines or pre-warming routing decisions.

---

## Upgrade notes

```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version   # should show 0.1.8
```

No adapter code changes required.

---

## Suggested ov_server integrations (optional, not blocking)

- Log `decision.estimated_tokens` to the routing audit log alongside model_id.
- Add `cost_per_1k_tokens` to `config.yaml` for OVH models; read from
  `decision` via `config.task_classes[decision.task_class].models` lookup for
  per-request cost estimation.
- Use `decide(request, trace=True)` in debug mode to populate `X-Routing-Decision`
  header with `decision.trace.scope_source` and `decision.trace.eliminated`.
