# Round 2 — ov_server integration feedback
**Against:** infergate v0.1.2
**Date:** 2026-05-14
**Integration goal this round:** Full wiring — OVServerBackend + OVEmbeddingProvider adapters,
Router.decide() replacing the router.py three-function chain, smoke tests on all live paths.

---

## What worked — do not regress

- `Router(config, backends, embedding_provider)` constructor signature is clean. No surprises.
- `await router.load_embeddings()` — single call, no configuration beyond passing the provider. Good.
- `RouteDecision` fields (`task_class`, `model_id`, `strategy`, `confidence`, `embedding`) are
  exactly what ov_server needs. Having `embedding` on the decision object saves a redundant encode
  call for the DB audit log.
- `strategy.value` — the `.value` on the `RoutingStrategy` enum works cleanly; no import needed
  for the string representation.
- `signal_only: true` correctly causes `load_embeddings()` to skip centroid computation for
  vision and web_search. No centroid file written, no error, clean startup log.
- `InferRequest(messages=..., tools=...)` — minimal constructor, no mandatory fields beyond
  messages. Important for the synthetic-tool workaround described below.
- `Backend` and `EmbeddingProvider` as `@runtime_checkable` Protocols — no base class needed.
  `OVServerBackend` and `OVEmbeddingProvider` satisfied the check without touching infergate source.
- `RouterConfig.from_dict()` backward-compat shims (`max_context_tokens→ctx_limit`,
  `provider→backend`, `embedding_threshold→embedding_min_confidence`) — no migration cost.

---

## API friction — harder than it should be

### 1. No way to register an external backend without implementing `chat()`

`OVServerBackend.chat()` raises `NotImplementedError` because ov_server handles inference —
infergate is routing-only here. This works fine, but the Protocol requires `chat()` to be
present. There is no `RoutingOnlyBackend` marker or `routing_only=True` flag on `Backend` to
signal intent and suppress the implicit contract that `chat()` will be called.

Current workaround: raise `NotImplementedError` with a docstring explaining why. Functional but
slightly awkward — a reader unfamiliar with the integration sees a `NotImplementedError` and
wonders if something is broken.

### 2. Cloud directive / multi-backend scoping is outside infergate

ov_server supports an `#ovh` / `#cloud` text directive that expands model scope to include OVH
remote models. infergate only has the local backend registered, so when this directive fires,
ov_server falls back to the old `router._select_model(scope_override="local+ovh")`. infergate
decides the `task_class`, but model selection reverts to legacy code.

This is by design (OVH is ov_server-specific), but the result is two routing paths in the same
function: infergate for normal requests, legacy `_select_model` for cloud directive. The seam
is visible in the code and requires keeping `router._select_model()` alive for this one case.

No proposed change — this is an ov_server deployment concern, not a library gap. Documenting it
so the infergate session understands why `router._select_model` is not yet deleted.

### 3. `load_embeddings()` runs even when no task class uses embedding routing

At startup, `await _ig_router.load_embeddings()` is called unconditionally. If all task classes
were `signal_only: true` there would be nothing to compute, but the call still happens. Minor —
in ov_server there are always embedding-routed classes — but worth noting for future callers
with pure-signal configs.

### 4. Duplicate centroid computation during transition

Both `router._load_embedding_centroids()` (legacy) and `_ig_router.load_embeddings()` (infergate)
run at startup and compute centroids using the same embedding model. This is a transitional state
(the legacy function also ensures `emb_model` is in globals before the provider accesses it),
but it means two embedding encode passes over the same class labels at startup.

Not a correctness issue. Will be resolved when the legacy centroid loader is deleted and startup
order is adjusted so the embedding model is guaranteed loaded before `Router` is constructed.

---

## Missing — needed but absent

### 1. `Backend` routing-only mode marker

A way to declare that a backend is routing-only and will never have `chat()` called. Options:
- A class attribute: `routing_only: bool = False` on `Backend` Protocol
- A subprotocol: `RoutingBackend` that omits `chat()`
- A flag in `RouterConfig` per backend: `mode: routing_only`

Use case: infergate is deployed as a pure router in front of an existing inference stack
(ov_server, vLLM, llama.cpp). The backend exists only to report `available_models()` and
`loaded_model_ids()` — `chat()` is never the exit point. Making this a first-class concept
would eliminate the `NotImplementedError` pattern and make the architecture self-documenting.

### 2. `force_tier` field on `InferRequest` not used — is it load-bearing?

`InferRequest` has a `force_tier` field. ov_server does not use it (task directives like `#code`
are expressed via message content and handled by infergate's signal detection). It's unclear
whether `force_tier` is a mechanism ov_server should use instead of directives, or whether it
is intended for a different caller pattern. A short docstring or example would remove the
ambiguity.

---

## Proposed changes with rationale

### P1 — Add `routing_only: ClassVar[bool] = False` to `Backend` Protocol

```python
class Backend(Protocol):
    is_local: ClassVar[bool]
    routing_only: ClassVar[bool]   # new — default False; True = chat() will never be called
    def name(self) -> str: ...
    def available_models(self) -> list[str]: ...
    def loaded_model_ids(self) -> list[str]: ...
    async def chat(self, request: InferRequest, model_id: str) -> dict: ...
```

`OVServerBackend` would set `routing_only = True`. `Router.decide()` could log a warning if
`routing_only` is False but `chat()` raises `NotImplementedError`. No behaviour change for
standard backends.

### P2 — Document `force_tier` with a one-line docstring or example in `InferRequest`

No code change needed — just clarify when a caller should use `force_tier` vs. putting a
task directive in the message content. The current ambiguity makes it hard to know whether
ov_server is missing a feature by not using it.

---

## Explicit non-requests

- **Do not add OVH / multi-backend scoping to infergate.** The cloud directive is an ov_server
  deployment concept tied to a specific OVH account and HTTP proxy. It must stay in ov_server.
- **Do not add `openvino_genai`, `OVModelForFeatureExtraction`, or `optimum-intel` as
  dependencies or examples.** infergate must remain inference-stack-agnostic.
- **Do not add a `centroid_cache` or startup optimisation for the embedding pass.** That is
  ov_server's concern and will be handled by adjusting the startup order locally.
- **Do not expose `OVServerBackend` or `OVEmbeddingProvider` in the infergate library.**
  They live in `/opt/ov_server/infergate/` and are ov_server-private adapters.

---

## Upgrade delta

**From round 1 (v0.1.1 → v0.1.2):**
- P0 stale wheel — RESOLVED. 0.1.2 wheel installed cleanly; all 8 gap fields present.
- `signal_only` on `TaskClassConfig` — CONFIRMED working. Vision and web_search skip centroid.
- `RouterConfig.from_dict()` shims — CONFIRMED working. No config migration required.
- `modality` field on `ModelDescriptor` — CONFIRMED working. VLM models parsed correctly.

All round 1 blockers resolved. Round 2 integration completed successfully.
