# Response to Round 2 — infergate v0.1.2 → v0.1.3
**Date:** 2026-05-14
**Addresses:** `round_02_v0.1.2.md`
**Published:** infergate v0.1.3 on PyPI

---

## Done

- **P1 — `routing_only` property on `Backend` Protocol:** Added `@property def routing_only(self) -> bool: ...`
  to the `Backend` Protocol in `protocols.py`. Added `routing_only = False` to both concrete backends
  (`OpenAICompatBackend`, `OllamaBackend`) and to `MockBackend` in tests. `OVServerBackend` can now
  declare `routing_only = True` to make the routing-only architecture self-documenting, eliminating
  the `NotImplementedError` ambiguity.

- **P2 — `force_tier` clarified:** Comment updated from "override profile preference; skips complexity
  promotion" to "programmatic tier override (e.g. admin/assessor); use message directives (#code etc.)
  for user-initiated routing". This removes the ambiguity about when a caller should use `force_tier`
  vs. putting a task directive in the message content.

## Skipped — with reason

- **No `RoutingOnlyBackend` subprotocol or `routing_only` flag in `RouterConfig`:** The property
  approach on `Backend` directly matches the round 2 P1 proposal and is the minimal change. The
  explicit non-requests (no OVH scoping, no new router modes) are respected.

- **API friction items 2–4 (cloud directive seam, unconditional `load_embeddings()`, dual centroid
  computation):** All noted as "by design" or "transitional state" in the round file with no
  proposed change. No action taken.

## Breaking changes

none — `routing_only` is a new property. Existing concrete backends that don't implement it will
fail `isinstance(obj, Backend)` only if `Backend` is used as a `@runtime_checkable` Protocol check.
`MockBackend` and both bundled backends now implement it. Any third-party backend must add
`routing_only = False` (or `True` if routing-only) to remain Protocol-compliant.

## Upgrade notes

```bash
source /home/jerzy/ov_env/bin/activate && pip install --upgrade infergate
pip show infergate | grep Version   # expect: 0.1.3
```

**Adapter update required:** Add `routing_only = True` to `OVServerBackend` and
`routing_only = False` to `OVEmbeddingProvider` is not needed (EmbeddingProvider Protocol
was not changed).
