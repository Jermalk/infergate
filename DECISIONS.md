# DECISIONS.md

### 2026-05-12 — ov_server reconnection is out of scope for this repo

**Decision:** Phase 2 (reconnecting ov_server to use infergate) will be done in a separate Claude Code session scoped to the ov_server directory, treating infergate as a PyPI dependency.

**Rationale:** Keeps the two codebases cleanly separated. infergate has no runtime knowledge of ov_server internals. ov_server reconnection is an integration concern, not an infergate concern. A dedicated session with ov_server as root prevents ov_server-specific implementation details from leaking into infergate's design.

**Rejected alternative:** Developing the reconnection inside this project with direct imports of ov_server internals.

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
