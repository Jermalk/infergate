# DECISIONS.md

### 2026-05-12 — ov_server reconnection is out of scope for this repo

**Decision:** Phase 2 (reconnecting ov_server to use infergate) will be done in a separate Claude Code session scoped to the ov_server directory, treating infergate as a PyPI dependency.

**Rationale:** Keeps the two codebases cleanly separated. infergate has no runtime knowledge of ov_server internals. ov_server reconnection is an integration concern, not an infergate concern. A dedicated session with ov_server as root prevents ov_server-specific implementation details from leaking into infergate's design.

**Rejected alternative:** Developing the reconnection inside this project with direct imports of ov_server internals.

### 2026-05-14 — User policy object rejected

**Decision:** No user policy object added to infergate. Per-user routing constraints (scope caps, tier limits) are the caller's responsibility, expressed by choosing whether to call `reselect()` and what parameters to pass.

**Rationale:** infergate is a routing engine, not an authorization layer. Every policy dimension (scope, tier) is already controllable by the caller through existing API surface. A policy object would duplicate RouterConfig fields, require precedence rules for every combination, and drag identity/auth concepts into the library.

**Rejected alternative:** `InferRequest.scope_override` and `InferRequest.max_tier` fields as a lightweight policy carrier.

**Affects:** `src/infergate/types.py`, `src/infergate/router.py`

### 2026-05-14 — Embed cache stores full routing result tuple

**Decision:** `_EmbedCache` stores the full `(task_class, confidence, embedding)` tuple returned by `route_by_embedding()`, not just the raw embedding vector.

**Rationale:** Simpler implementation — no need to expose or duplicate the centroid comparison logic. Centroids are stable for the Router's lifetime; if they change, the Router is typically recreated. The practical use case (throughput optimization for repeated queries) is fully served.

**Rejected alternative:** Cache only the raw embedding vector and re-run centroid comparison on each cache hit, which would be more correct if centroids change at runtime but adds complexity with no real-world benefit.

**Affects:** `src/infergate/router.py` (`_EmbedCache`, `_classify_vec`)

**Affects:** CLAUDE.md scope, PROGRESS.md next actions, what gets committed to this repo.

---

### 2026-05-12 — Scope naming uses local/remote/hybrid not ov_server strings

**Decision:** infergate uses `"local"` / `"remote"` / `"hybrid"` scope values with an `is_local: bool` flag on each Backend, rather than ov_server's `"local"` / `"local+ovh"` / `"all"` string-contains semantics.

**Rationale:** The ov_server scheme couples scope names to provider names and uses string-contains matching which is fragile. The `is_local` flag is explicit and backend-agnostic.

**Rejected alternative:** Preserving ov_server's `"local+ovh"` / `"all"` strings to ease migration.

**Affects:** `src/infergate/config.py` `RouterConfig.from_dict()`, `src/infergate/selector.py` `_scope_allows()`.

---

### 2026-05-12 — InferRequest includes tools field

**Decision:** `InferRequest` has a `tools: list[dict] | None = None` field even though concept.md omits it.

**Rationale:** The routing flowchart explicitly includes `has_tools` as a signal detector trigger. Without it, web_search task class can never be reached via signal detection.

**Rejected alternative:** Omitting tools and only routing web_search via keyword config.

**Affects:** `src/infergate/types.py`, `src/infergate/signals.py`.

---

### 2026-05-12 — TaskClassConfig includes examples field

**Decision:** `TaskClassConfig` has `examples: list[str]` for supplementing the description when computing centroids.

**Rationale:** ov_server uses examples to build better centroids (more representative embedding). The concept spec omits the field but the underlying logic supports it.

**Rejected alternative:** Description-only centroids as per the concept spec.

**Affects:** `src/infergate/config.py`, `src/infergate/embeddings.py`.

---

### 2026-05-12 — pydantic-settings for gateway operational config

**Decision:** `demo/gateway.py` uses `pydantic-settings` `BaseSettings` with `env_prefix="INFERGATE_"` for all operational parameters (URLs, API keys, log level, config path).

**Rationale:** Replaces six scattered `os.environ.get()` calls with a single typed, validated, documented class. `SecretStr` for API keys prevents token leakage in logs and repr. Supports `.env` files for local dev. Directly maps to Kubernetes Secret + ConfigMap pattern.

**Rejected alternative:** Bare `os.environ.get()` with inline defaults — no validation, no type safety, no `.env` support.

**Affects:** `demo/gateway.py`, `pyproject.toml` `[demo]` extras.

---

### 2026-05-12 — Settings class lives in gateway only, not in the library

**Decision:** `Settings(BaseSettings)` is defined in `demo/gateway.py` and is not exposed as part of the `infergate` library.

**Rationale:** Operational configuration (ports, credentials, infrastructure URLs) is an application-layer concern. The library's config boundary is `RouterConfig.from_dict(dict)` — how that dict is sourced is the caller's responsibility. Adding `BaseSettings` to the library would couple it to pydantic-settings and impose opinions on config loading that library users may not share.

**Rejected alternative:** An `infergate.settings` module exposing a base `Settings` class for gateway implementors.

**Affects:** `demo/gateway.py`, `src/infergate/` (unchanged).

---

### 2026-05-12 — Routing config YAML-only; no env var overrides for thresholds

**Decision:** Routing topology (task classes, model lists, thresholds, keywords) is expressed exclusively in `config.yaml`. Environment variables control only operational parameters (URLs, keys, log level, config path).

**Rationale:** Routing topology is structural configuration that belongs in version control alongside the code. Allowing threshold overrides via env vars would make the effective routing config invisible — split across a file and environment state that may differ between instances. ConfigMaps in Kubernetes handle this correctly: topology is a ConfigMap, secrets are env vars.

**Rejected alternative:** `INFERGATE_EMBEDDING_THRESHOLD` and similar env vars that override individual `RouterSettings` fields.

**Affects:** `demo/gateway.py` `Settings`, `demo/config.yaml`.

---

### 2026-05-14 — sentence-transformers moved to optional [local-embed] extra

**Decision:** `sentence-transformers` is not a core dependency; users install it via `pip install infergate[local-embed]` only if they use `SentenceTransformerProvider`.

**Rationale:** sentence-transformers pulls the full torch stack (~2 GB CPU wheels). Library users who inject a custom `EmbeddingProvider` should not pay that cost. The import is already lazy inside `_load()`, so no import-time breakage.

**Rejected alternative:** Keeping it mandatory — every library install forces a torch download regardless of usage.

**Affects:** `pyproject.toml` `[project.optional-dependencies]`.

---

### 2026-05-14 — pydantic and pyyaml removed from core dependencies

**Decision:** `pydantic` and `pyyaml` moved from `[project.dependencies]` to the `[demo]` optional extra.

**Rationale:** Neither is imported anywhere in `src/infergate/`. Both are used exclusively by `demo/gateway.py`. Including them in core deps forced ~15 MB of unnecessary installs on all library users.

**Rejected alternative:** Keeping them in core for "convenience" — there is no convenience if the library never imports them.

**Affects:** `pyproject.toml`.

---

### 2026-05-14 — Gap list gaps 2–8 closed; signal_only replaces hardcoded name set

**Decision:** `TaskClassConfig.signal_only: bool` replaces the hardcoded `_SIGNAL_ONLY_CLASSES` frozenset. Any task class that should never appear as an embedding target is marked at config level.

**Rationale:** Hardcoding class names couples the library to deployment-specific naming. Any user who names their VLM class `"image_tasks"` instead of `"vision"` would silently get wrong centroids.

**Rejected alternative:** Keep `_SIGNAL_ONLY_CLASSES` and document that users must match those exact strings.

**Affects:** `src/infergate/config.py`, `src/infergate/embeddings.py`, `src/infergate/signals.py`.

---

### 2026-05-14 — Task-class directives compiled from config keys at Router init

**Decision:** `Router.__init__` compiles `self._task_directive_re` from `config.task_classes.keys()` once at startup. `task_class_directive()` accepts an optional pre-compiled `pattern`; falls back to the hardcoded legacy set when called without one.

**Rationale:** A user who adds a `"sql"` task class should be able to use `#sql` without patching the library. The fallback preserves backward compatibility for code that calls `task_class_directive` directly in tests.

**Rejected alternative:** Keep `_TASK_DIRECTIVE_RE = re.compile(r'#(code|document|general)\b')` and require users to monkey-patch.

**Affects:** `src/infergate/signals.py`, `src/infergate/router.py`.

---

### 2026-05-14 — NoModelAvailable raised instead of empty-string sentinel

**Decision:** `select_model` raises `NoModelAvailable(task_class, scope)` when no backend is reachable. The exception carries `task_class` and `scope` as attributes. The old `return ("", "", False)` path is removed.

**Rationale:** Silent empty-string returns propagate invisibly — callers that forget to check send a request with no model ID. A typed exception forces explicit handling and produces actionable error messages.

**Rejected alternative:** Add an `error: str | None` field to `RouteDecision` (Option B from the gap analysis). Rejected because it couples the router to deployment-specific fallback knowledge.

**Affects:** `src/infergate/types.py`, `src/infergate/selector.py`, `src/infergate/__init__.py`.

---

### 2026-05-14 — Modality field on ModelDescriptor for VLM routing

**Decision:** `ModelDescriptor.modality: Modality = "text"` where `Modality = Literal["text", "vision", "any"]`. `select_model` accepts `required_modality` and filters the candidate pool. Router derives `required_modality="vision"` from `has_images()`.

**Rationale:** Without modality, a loaded LLM can satisfy a vision routing request because the selector sees it as available. ov_server learned this the hard way (separate `loaded_vlm_models` dict).

**Rejected alternative:** Making `Backend.loaded_model_ids()` return `dict[str, Modality]` — breaking change deferred to 0.2.x.

**Affects:** `src/infergate/config.py`, `src/infergate/selector.py`, `src/infergate/router.py`.

---

### 2026-05-14 — routing_only property on Backend Protocol (round 2 feedback)

**Decision:** `Backend` Protocol gains `@property def routing_only(self) -> bool: ...`. Concrete backends return `False`; a routing-only adapter (e.g. `OVServerBackend`) returns `True`. No router behaviour changes — the property is informational and self-documenting.

**Rationale:** ov_server deploys infergate as a pure router: `chat()` is never called, so implementing it with `raise NotImplementedError` looked broken to unfamiliar readers. A first-class `routing_only` property declares intent explicitly without adding a subprotocol or config flag.

**Rejected alternative:** `RoutingBackend` subprotocol omitting `chat()` — would require isinstance checks against a second Protocol and complicates the type hierarchy for minor gain.

**Affects:** `src/infergate/protocols.py`, `src/infergate/backends/openai_compat.py`, `src/infergate/backends/ollama.py`, `tests/conftest.py`.

---

### 2026-05-14 — RouteStrategy.KEYWORD separated from SIGNAL

**Decision:** Hashtag directives (`#code`, `#document`, `#general`) produce `RouteStrategy.KEYWORD`, while image detection, tools, long context, and keyword phrase matching produce `RouteStrategy.SIGNAL`. `task_class_directive()` is called by `Router.decide()` before `detect_signal()`.

**Rationale:** The spec defines KEYWORD for explicit user intent and SIGNAL for objective evidence. Collapsing both into SIGNAL makes the strategy field less useful for monitoring — you cannot tell from the header whether routing was user-directed or automatically inferred. Separating them also keeps `detect_signal()` single-responsibility: it covers objective signals only.

**Rejected alternative:** Having `detect_signal()` return the directive and set SIGNAL for everything, as was the original implementation.

**Affects:** `src/infergate/router.py`, `src/infergate/signals.py`, `src/infergate/types.py`.
