"""
Model selection: complexity scoring and tier-based pick with prefer-loaded logic.
"""
import logging
import re

from infergate.config import ModelDescriptor
from infergate.config import RouterConfig
from infergate.protocols import Backend
from infergate.signals import text_content


log = logging.getLogger("infergate")

_COMPLEXITY_SIGNALS: tuple[str, ...] = (
    "analyze", "compare", "explain in detail", "evaluate", "critique",
    "summarize", "translate", "implement", "design", "architecture",
    "step by step", "in depth", "thoroughly", "comprehensive", "detailed",
)
_SIMPLE_Q_RE = re.compile(
    r"^(what|who|when|where|how much|how many|is|are|was|were|can|does|do|did)"
    r"\b.{0,60}\??\s*$",
    re.IGNORECASE,
)


def complexity_score(messages: list[dict]) -> float:
    """0.0 = simple, 1.0 = complex. Used to break ties within a preference tier."""
    last_user = next(
        (text_content(m) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    words = last_user.split()
    score = 0.0
    if len(words) > 50:
        score += 0.3
    if len(words) > 150:
        score += 0.2
    hits = sum(1 for s in _COMPLEXITY_SIGNALS if s in last_user.lower())
    score += min(hits * 0.15, 0.4)
    if sum(1 for m in messages if m.get("role") == "user") > 4:
        score += 0.1
    if _SIMPLE_Q_RE.match(last_user.strip()):
        score -= 0.3
    return max(0.0, min(1.0, score))


def _scope_allows(backend: Backend, scope: str) -> bool:
    if scope == "local":
        return backend.is_local
    if scope == "remote":
        return not backend.is_local
    return True  # "hybrid"


def select_model(
    task_class: str,
    config: RouterConfig,
    backends: dict[str, Backend],
    effective_scope: str,
    profile_pref: str,
    complexity: float = 0.0,
    estimated_tokens: int = 0,
) -> tuple[str, str, bool]:
    """Return (backend_name, model_id, prefer_loaded).

    Selection order:
      1. Scope filter   — eliminate backends not eligible under effective_scope
      2. Context filter — skip models whose ctx_limit < estimated_tokens
      3. Prefer-loaded  — for "fastest" pref, prefer warm fast-tier models first
      4. Tier pick      — fastest → balanced → best
      5. Complexity     — "balanced" + complexity > 0.65 promotes to "best"

    Falls back to the "general" task class when task_class has no config entry.
    Returns ("", "", False) only when no backend at all is reachable; the caller
    is expected to translate this into a 503 / unavailable error.
    """
    cls_cfg = config.task_classes.get(task_class) or config.task_classes.get("general")
    if cls_cfg is None:
        return _fallback(backends, effective_scope)
    all_models: list[ModelDescriptor] = cls_cfg.models

    available: list[ModelDescriptor] = []
    for m in all_models:
        backend = backends.get(m.backend)
        if backend is None:
            continue
        if not _scope_allows(backend, effective_scope):
            continue
        if m.id not in backend.available_models():
            log.warning("'%s' not available on backend '%s' — skipped", m.id, m.backend)
            continue
        if estimated_tokens and m.ctx_limit and estimated_tokens > m.ctx_limit:
            log.info("'%s' ctx_limit %d < prompt ~%d — skipped", m.id, m.ctx_limit, estimated_tokens)
            continue
        available.append(m)

    if not available:
        return _fallback(backends, effective_scope)

    pref = profile_pref
    if pref == "balanced" and complexity > 0.65:
        pref = "best"

    all_loaded: set[str] = set()
    for b in backends.values():
        all_loaded.update(b.loaded_model_ids())

    def _fastest(pool: list[ModelDescriptor]) -> ModelDescriptor | None:
        return next((m for m in pool if m.tier == "fast"), None)

    def _balanced(pool: list[ModelDescriptor]) -> ModelDescriptor | None:
        # Prefer a balanced-tier local model; fall back to any local model.
        local_balanced = [m for m in pool if backends[m.backend].is_local and m.tier == "balanced"]
        if local_balanced:
            return local_balanced[-1]
        local = [m for m in pool if backends[m.backend].is_local]
        return local[-1] if local else None

    def _best(pool: list[ModelDescriptor]) -> ModelDescriptor | None:
        best = [m for m in pool if m.tier == "best"]
        if best:
            return best[-1]
        return pool[-1] if pool else None

    def _pick(pool: list[ModelDescriptor]) -> ModelDescriptor | None:
        if pref == "fastest":
            return _fastest(pool) or _balanced(pool) or _best(pool)
        if pref == "balanced":
            return _balanced(pool) or _best(pool)
        return _best(pool)

    chosen: ModelDescriptor | None
    prefer_loaded = False
    if pref == "fastest":
        loaded_fast = [m for m in available if m.id in all_loaded and m.tier == "fast"]
        chosen = _pick(loaded_fast)
        if chosen:
            prefer_loaded = True
        else:
            chosen = _pick(available)
    else:
        chosen = _pick(available)

    if chosen is None:
        return _fallback(backends, effective_scope)

    return (chosen.backend, chosen.id, prefer_loaded)


def _fallback(backends: dict[str, Backend], scope: str) -> tuple[str, str, bool]:
    for b in backends.values():
        if not _scope_allows(b, scope):
            continue
        models = b.available_models()
        if models:
            log.error("select_model fallback — using first available '%s' on '%s'", models[0], b.name())
            return (b.name(), models[0], False)
    log.error("select_model fallback — no backends available for scope '%s'", scope)
    return ("", "", False)
