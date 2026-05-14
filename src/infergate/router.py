"""
Router — public entry point. Wires signal detection, embedding routing, and model selection.
"""
import logging
import re
import time
from collections import OrderedDict

import numpy as np

from infergate.config import RouterConfig
from infergate.embeddings import compute_centroids
from infergate.embeddings import route_by_embedding
from infergate.protocols import Backend
from infergate.protocols import EmbeddingProvider
from infergate.selector import complexity_score
from infergate.selector import select_model
from infergate.signals import detect_signal
from infergate.signals import has_cloud_directive
from infergate.signals import has_images
from infergate.signals import last_user_text
from infergate.signals import task_class_directive
from infergate.signals import text_content
from infergate.types import EliminatedCandidate
from infergate.types import InferRequest
from infergate.types import RouteDecision
from infergate.types import RouteStrategy
from infergate.types import RouteTrace


log = logging.getLogger("infergate")

_EmbedResult = tuple[str, float, "list[float] | None"]


class _EmbedCache:
    """LRU cache for route_by_embedding() results. Single-threaded / single event loop."""

    def __init__(self, maxsize: int) -> None:
        self._maxsize = maxsize
        self._data: OrderedDict[str, _EmbedResult] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> _EmbedResult | None:
        if self._maxsize == 0 or key not in self._data:
            self._misses += 1
            return None
        self._data.move_to_end(key)
        self._hits += 1
        return self._data[key]

    def put(self, key: str, value: _EmbedResult) -> None:
        if self._maxsize == 0:
            return
        if key in self._data:
            self._data.move_to_end(key)
        else:
            if len(self._data) >= self._maxsize:
                self._data.popitem(last=False)
        self._data[key] = value

    def __len__(self) -> int:
        return len(self._data)


class Router:
    def __init__(
        self,
        config: RouterConfig,
        backends: dict[str, Backend],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._config = config
        self._backends = backends
        self._provider = embedding_provider
        self._centroids: dict[str, np.ndarray] = {}
        self._embed_cache = _EmbedCache(config.router.embedding_cache_size)
        task_names = sorted(config.task_classes.keys())
        self._task_directive_re: re.Pattern | None = re.compile(
            r'#(' + '|'.join(re.escape(k) for k in task_names) + r')\b',
            re.IGNORECASE,
        ) if task_names else None

    @classmethod
    def from_config(
        cls,
        config: dict | RouterConfig,
        backends: dict[str, Backend],
        embedding_provider: EmbeddingProvider | None = None,
    ) -> "Router":
        if isinstance(config, dict):
            config = RouterConfig.from_dict(config)
        return cls(config, backends, embedding_provider)

    async def load_embeddings(self) -> None:
        """Embed all task class descriptions and store as centroids.

        Must be awaited before the first decide() call.
        No-op when no EmbeddingProvider is configured.
        """
        if self._provider is None:
            log.warning("[router] no EmbeddingProvider — embedding routing disabled")
            return
        try:
            self._centroids = await compute_centroids(self._config.task_classes, self._provider)
            log.info("[router] centroids ready for: %s", list(self._centroids))
        except Exception as exc:
            log.warning("[router] centroid computation failed (%s) — embedding routing disabled", exc)
            self._centroids = {}

    def cache_stats(self) -> dict[str, int]:
        """Embedding cache hit/miss counters and current occupancy.

        Counters reset on Router construction. Use to tune embedding_cache_size
        without enabling per-request trace overhead.
        Returns: {"hits", "misses", "size", "capacity"}
        """
        c = self._embed_cache
        return {"hits": c._hits, "misses": c._misses, "size": len(c), "capacity": c._maxsize}

    def _classify_vec(self, vec_list: list[float]) -> _EmbedResult:
        """Centroid comparison for a pre-encoded vector. Does not call the provider."""
        vec = np.array(vec_list, dtype=float)
        norm = np.linalg.norm(vec)
        vec = vec / max(norm, 1e-9)
        best_class, best_score = "general", 0.0
        for tc, centroid in self._centroids.items():
            score = float(np.dot(vec, centroid))
            if score > best_score:
                best_class, best_score = tc, score
        if best_score < self._config.router.embedding_min_confidence:
            best_class = "general"
        return (best_class, best_score, vec.tolist())

    async def decide(self, request: InferRequest, *, trace: bool = False) -> RouteDecision:
        """Main routing entry point. Returns RouteDecision without executing the request.

        Pipeline:
          1. Directive check  — #code / #document / #general hashtag → RouteStrategy.KEYWORD
          2. Signal detection — image / tools / long-context / keyword → RouteStrategy.SIGNAL
          3. Embedding routing — cosine similarity against task-class centroids
          4. Scope resolution — per-class override > #ovh/#cloud directive > global scope
          5. Model selection  — prefer loaded → tier → complexity → context limit check

        Pass trace=True to populate RouteDecision.trace with elimination reasons and timing.
        """
        settings = self._config.router
        eliminated: list[EliminatedCandidate] | None = [] if trace else None

        # ── Stage 1 & 2: signal detection ─────────────────────────────────────
        # Directive check is separated from detect_signal so we can assign the
        # correct RouteStrategy (KEYWORD vs SIGNAL) without detect_signal needing
        # to know about strategy types.
        images_present = has_images(request.messages)
        directive = task_class_directive(request.messages, self._task_directive_re)
        if directive:
            task_class: str | None = directive
            strategy: RouteStrategy | None = RouteStrategy.KEYWORD
        else:
            signal_class = detect_signal(request, settings, images_present=images_present)
            task_class = signal_class
            strategy = RouteStrategy.SIGNAL if signal_class else None
        confidence = 1.0
        embedding: list[float] | None = None
        embedding_ms: float | None = None
        cache_hit: bool | None = None

        # ── Stage 3: embedding classification (slow path) ─────────────────────
        if task_class is None:
            if self._provider and self._centroids:
                query = last_user_text(request.messages)
                cached = self._embed_cache.get(query)
                if cached is not None:
                    task_class, confidence, embedding = cached
                    cache_hit = True
                else:
                    _t0 = time.perf_counter() if trace else 0.0
                    task_class, confidence, embedding = await route_by_embedding(
                        query,
                        self._centroids,
                        self._provider,
                        settings.embedding_min_confidence,
                    )
                    if trace:
                        embedding_ms = (time.perf_counter() - _t0) * 1000
                    self._embed_cache.put(query, (task_class, confidence, embedding))
                    cache_hit = False
                strategy = (
                    RouteStrategy.EMBEDDING
                    if confidence >= settings.embedding_min_confidence
                    else RouteStrategy.FALLBACK
                )
            else:
                task_class = "general"
                confidence = 0.0
                strategy = RouteStrategy.FALLBACK

        # ── Stage 4: scope resolution ──────────────────────────────────────────
        cls_cfg = self._config.task_classes.get(task_class)
        if cls_cfg and cls_cfg.scope_override:
            effective_scope = cls_cfg.scope_override
            scope_source = "class_override"
        elif has_cloud_directive(request.messages):
            effective_scope = "remote"
            scope_source = "cloud_directive"
        else:
            effective_scope = self._config.provider_scope
            scope_source = "global"

        # ── Stage 5: model selection ───────────────────────────────────────────
        profile = self._config.profiles.get(self._config.active_profile, {})
        profile_pref = profile.get("model_preference", "balanced")

        # Token estimate uses the same text_content() as signal detection for
        # consistency; multimodal messages (list content) are handled correctly.
        total_tokens = sum(
            len(text_content(m)) for m in request.messages
        ) // 4

        required_modality = "vision" if images_present else None
        backend_name, model_id, prefer_loaded, estimated_cost_usd = select_model(
            task_class=task_class,
            config=self._config,
            backends=self._backends,
            effective_scope=effective_scope,
            profile_pref=profile_pref,
            complexity=complexity_score(request.messages),
            estimated_tokens=total_tokens,
            force_tier=request.force_tier,
            required_modality=required_modality,
            _eliminated=eliminated,
        )

        route_trace = RouteTrace(
            eliminated=eliminated,
            scope_source=scope_source,
            embedding_ms=embedding_ms,
            cache_hit=cache_hit,
        ) if trace else None

        return RouteDecision(
            backend=backend_name,
            model_id=model_id,
            task_class=task_class,
            strategy=strategy,
            confidence=confidence,
            prefer_loaded=prefer_loaded,
            embedding=embedding,
            task_directive=directive,
            estimated_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
            trace=route_trace,
        )

    async def decide_batch(
        self,
        requests: list[InferRequest],
        trace: bool = False,
    ) -> list[RouteDecision]:
        """Route a batch of requests sharing a single embed_batch() call.

        Signal/keyword detection runs per-request (O(1)). Remaining requests are
        checked against the embedding cache; uncached unique queries are collected
        and sent as one embed_batch() call. Results are stored in the cache before
        routing completes, so a subsequent decide() or decide_batch() for the same
        queries will hit the cache.

        embedding_ms is not set in trace for batch requests — the embed_batch() call
        is shared and its wall time is not attributable to individual requests.
        """
        if not requests:
            return []

        settings = self._config.router
        profile = self._config.profiles.get(self._config.active_profile, {})
        profile_pref = profile.get("model_preference", "balanced")

        # ── Phase 1: signal detection + cache lookup ───────────────────────────
        task_classes:  list[str | None]           = []
        strategies:    list[RouteStrategy | None] = []
        directives:    list[str | None]           = []
        images_flags:  list[bool]                 = []
        queries:       list[str | None]           = []
        cache_hits:    list[bool | None]          = []
        eliminateds:   list[list | None]          = []

        embed_results: dict[str, _EmbedResult] = {}
        uncached_queries: list[str] = []
        seen_uncached: set[str] = set()

        for req in requests:
            eliminated = [] if trace else None
            images_present = has_images(req.messages)
            directive = task_class_directive(req.messages, self._task_directive_re)

            if directive:
                task_classes.append(directive)
                strategies.append(RouteStrategy.KEYWORD)
                queries.append(None)
                cache_hits.append(None)
            else:
                signal_class = detect_signal(req, settings, images_present=images_present)
                if signal_class:
                    task_classes.append(signal_class)
                    strategies.append(RouteStrategy.SIGNAL)
                    queries.append(None)
                    cache_hits.append(None)
                else:
                    task_classes.append(None)
                    strategies.append(None)
                    if self._provider and self._centroids:
                        query = last_user_text(req.messages)
                        cached = self._embed_cache.get(query)
                        if cached is not None:
                            embed_results[query] = cached
                            cache_hits.append(True)
                        else:
                            if query not in seen_uncached:
                                uncached_queries.append(query)
                                seen_uncached.add(query)
                            cache_hits.append(False)
                        queries.append(query)
                    else:
                        queries.append(None)
                        cache_hits.append(None)

            directives.append(directive)
            images_flags.append(images_present)
            eliminateds.append(eliminated)

        # ── Phase 2: one embed_batch() call for all uncached queries ───────────
        if uncached_queries and self._provider:
            vec_lists = await self._provider.embed_batch(
                [q[:2048] for q in uncached_queries]
            )
            for query, vec_list in zip(uncached_queries, vec_lists):
                result = self._classify_vec(vec_list)
                self._embed_cache.put(query, result)
                embed_results[query] = result

        # ── Phase 3: scope resolution + model selection per request ────────────
        decisions: list[RouteDecision] = []
        for i, req in enumerate(requests):
            task_class = task_classes[i]
            strategy   = strategies[i]
            directive  = directives[i]
            images_present = images_flags[i]
            query      = queries[i]
            cache_hit  = cache_hits[i]
            eliminated = eliminateds[i]
            confidence = 1.0
            embedding: list[float] | None = None

            if task_class is None:
                if query is not None and query in embed_results:
                    task_class, confidence, embedding = embed_results[query]
                else:
                    task_class = "general"
                    confidence = 0.0
                strategy = (
                    RouteStrategy.EMBEDDING
                    if confidence >= settings.embedding_min_confidence
                    else RouteStrategy.FALLBACK
                )

            cls_cfg = self._config.task_classes.get(task_class)
            if cls_cfg and cls_cfg.scope_override:
                effective_scope = cls_cfg.scope_override
                scope_source = "class_override"
            elif has_cloud_directive(req.messages):
                effective_scope = "remote"
                scope_source = "cloud_directive"
            else:
                effective_scope = self._config.provider_scope
                scope_source = "global"

            total_tokens = sum(len(text_content(m)) for m in req.messages) // 4
            required_modality = "vision" if images_present else None

            backend_name, model_id, prefer_loaded, estimated_cost_usd = select_model(
                task_class=task_class,
                config=self._config,
                backends=self._backends,
                effective_scope=effective_scope,
                profile_pref=profile_pref,
                complexity=complexity_score(req.messages),
                estimated_tokens=total_tokens,
                force_tier=req.force_tier,
                required_modality=required_modality,
                _eliminated=eliminated,
            )

            route_trace = RouteTrace(
                eliminated=eliminated,
                scope_source=scope_source,
                embedding_ms=None,
                cache_hit=cache_hit,
            ) if trace else None

            decisions.append(RouteDecision(
                backend=backend_name,
                model_id=model_id,
                task_class=task_class,
                strategy=strategy,
                confidence=confidence,
                prefer_loaded=prefer_loaded,
                embedding=embedding,
                task_directive=directive,
                estimated_tokens=total_tokens,
                estimated_cost_usd=estimated_cost_usd,
                trace=route_trace,
            ))

        return decisions

    def reselect(
        self,
        task_class: str,
        scope: str = "local",
        force_tier: str | None = None,
        complexity: float = 0.0,
        estimated_tokens: int = 0,
    ) -> RouteDecision:
        """Re-run model selection for an already-determined task_class with different constraints.

        Does not re-run signal detection or embedding routing — task_class must be supplied
        by the caller (typically taken from a prior RouteDecision.task_class).

        Scope values: "local" | "remote" | "local+remote"
        """
        profile = self._config.profiles.get(self._config.active_profile, {})
        profile_pref = profile.get("model_preference", "balanced")

        backend_name, model_id, prefer_loaded, estimated_cost_usd = select_model(
            task_class=task_class,
            config=self._config,
            backends=self._backends,
            effective_scope=scope,
            profile_pref=profile_pref,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            force_tier=force_tier,
        )

        return RouteDecision(
            backend=backend_name,
            model_id=model_id,
            task_class=task_class,
            strategy=RouteStrategy.RESELECT,
            confidence=1.0,
            prefer_loaded=prefer_loaded,
            estimated_cost_usd=estimated_cost_usd,
        )
