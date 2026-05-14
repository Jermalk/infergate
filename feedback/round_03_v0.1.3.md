# Round 3 — ov_server integration feedback
**Against:** infergate v0.1.3
**Date:** 2026-05-14
**Integration goal this round:** Identify residual friction after routing-only wiring is fully live.
Specifically: map every remaining router.py function that cannot yet be deleted and explain why.

---

## What worked — do not regress

- `routing_only = True` on `OVServerBackend` works exactly as intended. The attribute is visible
  at a glance; no `NotImplementedError` in the reader's path. P1 from round 2 fully resolved.
- `force_tier` docstring — the programmatic-vs-directive distinction is now clear.
  ov_server confirmed it does not need `force_tier` (directives via message content are enough).
- `signals.text_content()` and `has_images()` handle multimodal content (string + list) correctly.
  Vision routing fires on the first image-bearing message; no workaround needed.
- Silent skip of unregistered backends in `selector.select_model()` is the right default.
  ov_server lists `ovh` models in `config.yaml` task classes but does not register an OVH backend.
  infergate skips them cleanly and routes to the best available local model. Zero errors.
- `RouteDecision.embedding` field saves a redundant encode call for the DB audit log. Still used.

---

## API friction — harder than it should be

### 1. No public re-selection API → `router._select_model()` cannot be deleted

After `Router.decide()` returns a `RouteDecision`, an `#ovh` / `#cloud` directive in the message
requires re-selecting from the same `task_class` but with a wider scope (`local+ovh`) and a fixed
tier (`best`). infergate decides the `task_class` correctly, but there is no public `Router` method
to re-run model selection with different scope/tier parameters.

Current workaround (ov_server.py lines 1088–1098):

```python
if _cloud_directive:
    cplx = router.complexity_score(req)
    est_tokens = sum(len(_text_content(m)) for m in req.messages) // 4
    model_entry = router._select_model(
        task_class, active_profile_cfg, cplx, est_tokens,
        scope_override="local+ovh", pref_override="best",
    )
    model_id = model_entry["id"]
```

`router._select_model()` reads from `config.json` task_classes (not `config.yaml`), so it is a
parallel model catalogue duplicating what is already in infergate's config. This function cannot
be deleted until there is a library-side way to express: *"given this task_class, re-select a
model, but this time include the `ovh` backend and force tier to best."*

### 2. `RouteDecision` does not surface the detected task directive

`Router.decide()` detects `#code`, `#document`, `#general`, etc. directives internally, but the
`RouteDecision` does not carry which directive (if any) fired. ov_server needs this for the
routing audit log and for the response `X-Routing-Decision` header.

Current workaround (ov_server.py line 1102):

```python
_task_directive = router.task_class_directive(req.messages)
```

This uses a hardcoded `re.compile(r'#(code|document|general)\b')` — it does not cover `vision`,
`web_search`, or any future task class added to `config.yaml`. The infergate router detects
directives against the live task-class name set; the ov_server fallback is frozen at 3 classes.

### 3. `complexity_score` and `has_cloud_directive` duplicated in router.py

`router.complexity_score()` and `router._has_cloud_directive()` are near-identical copies of
`infergate.selector.complexity_score()` and `infergate.signals.has_cloud_directive()`. They
remain in router.py only because:

- `selector.complexity_score(messages: list[dict])` requires `list[dict]`, but the cloud directive
  path has a `ChatRequest` with Pydantic message objects (`m.role`, not `m["role"]`). Converting
  is trivial but was not done yet.
- `signals.has_cloud_directive()` is not exported from the top-level `infergate` module. It is
  accessible via `from infergate.signals import has_cloud_directive` but this import path is not
  documented as stable public API.

These are ov_server clean-up items, not blocking gaps. Documenting for completeness.

---

## Missing — needed but absent

### 1. `Router.reselect()` — post-decide scope-switch

A public method to re-run model selection given a `task_class` already decided by `Router.decide()`,
with caller-supplied scope and tier overrides:

```python
def reselect(
    self,
    task_class: str,
    scope: str = "local",          # "local" | "remote" | "local+remote"
    force_tier: str | None = None, # "fast" | "balanced" | "best"
    complexity: float = 0.0,
    estimated_tokens: int = 0,
) -> RouteDecision:
    ...
```

Use case: `#cloud` / `#ovh` directive fires. ov_server calls `Router.decide()` to get
`task_class`, then calls `Router.reselect(task_class, scope="local+remote", force_tier="best")`
to pick the best model including registered remote backends.

What this enables: OVH backend registered at startup as a second entry in the `backends` dict.
`reselect()` would pick from both local and remote descriptors. ov_server's `router._select_model`
and the parallel config.json task_classes could then be deleted.

**Constraint:** "remote" scope handling in `reselect()` must remain a pure selection call —
no HTTP requests, no health checks, no availability verification. ov_server verifies OVH
availability at the proxy level before forwarding.

### 2. `RouteDecision.task_directive: str | None`

Add the detected task directive (if any) to the returned decision:

```python
@dataclass
class RouteDecision:
    ...
    task_directive: str | None = None  # e.g. "code", "document", None
```

The router already has this value at the point `decide()` returns it; surfacing it costs nothing.
This lets callers remove their own directive-detection pass and ensures the reported directive
is consistent with the router's live task-class set (not a frozen regex).

---

## Proposed changes with rationale

### P1 — `Router.reselect(task_class, scope, force_tier, …) → RouteDecision`

Wraps `selector.select_model()` as a public method on `Router`. Takes a `task_class` string
already returned by `decide()`. Returns a new `RouteDecision` with `strategy=FORCED` (or a
new `RESELECT` strategy value) and `confidence=1.0`. Does not re-run signal detection or
embedding routing.

This unblocks full deletion of `router._select_model()` in ov_server, collapsing two routing
paths into one config source (`config.yaml`).

### P2 — `RouteDecision.task_directive: str | None = None`

No behaviour change. Router sets it to the matched directive name (e.g. `"code"`) or `None`
when no directive was present. Callers that do not need it ignore the field.

---

## Explicit non-requests

- **Do not add OVH-specific scope names or credentials to infergate.** `reselect()` should
  accept the generic strings `"local"`, `"remote"`, `"local+remote"` — the same strings already
  used in `selector._scope_allows()`. OVH is one possible remote backend; infergate must not know it.
- **Do not add health checks or latency probes to `reselect()`.** Selection is synchronous and
  pure; backend availability is the caller's responsibility.
- **Do not add `has_cloud_directive()` to the top-level `infergate` module unless it is part of
  a broader `signals` re-export.** A `from infergate.signals import has_cloud_directive` import
  is already usable; a top-level re-export is a nice-to-have, not a blocker.
- **Do not move ov_server adapter code into infergate.** `OVServerBackend` and
  `OVEmbeddingProvider` stay in `/opt/ov_server/infergate/` as ov_server-private adapters.

---

## Upgrade delta

**From round 2 (v0.1.2 → v0.1.3):**
- P1 `routing_only: bool` on `Backend` Protocol — CONFIRMED. `OVServerBackend.routing_only = True`
  is recognised; `chat()` is never called by the router. Architecture is now self-documenting.
- P2 `force_tier` docstring — CONFIRMED. Ambiguity resolved; ov_server uses message directives
  for user-initiated routing, `force_tier` reserved for programmatic overrides.
- Breaking change note in addressed file (third-party backends must add `routing_only = False`)
  — noted. ov_server is the only downstream; no impact.

All round 2 items resolved. Round 3 integration: routing-only wiring complete and stable.
Remaining router.py functions (`_select_model`, `complexity_score`, `_has_cloud_directive`,
`task_class_directive`) cannot be deleted until P1 and P2 above are available.
