"""
Router — public entry point. Wires signal detection, embedding routing, and model selection.
"""
import logging

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
from infergate.types import InferRequest
from infergate.types import RouteDecision
from infergate.types import RouteStrategy


log = logging.getLogger("infergate")


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

    async def decide(self, request: InferRequest) -> RouteDecision:
        """Main routing entry point. Returns RouteDecision without executing the request.

        Pipeline:
          1. Signal detection   — O(1)
          2. Embedding routing  — async, provider handles executor
          3. Scope resolution   — per-class override > cloud directive > global
          4. Model selection    — prefer loaded → tier → complexity → context check
        """
        settings = self._config.router

        # ── Stage 1: signal detection ──────────────────────────────────────
        signal_class = detect_signal(request, settings)
        strategy = RouteStrategy.SIGNAL if signal_class else None
        task_class = signal_class
        confidence = 1.0
        embedding: list[float] | None = None

        # ── Stage 2: embedding classification (slow path) ─────────────────
        if task_class is None:
            if self._provider and self._centroids:
                query = self._extract_query(request)
                task_class, confidence, embedding = await route_by_embedding(
                    query,
                    self._centroids,
                    self._provider,
                    settings.embedding_min_confidence,
                )
                strategy = (
                    RouteStrategy.EMBEDDING
                    if confidence >= settings.embedding_min_confidence
                    else RouteStrategy.FALLBACK
                )
            else:
                task_class = "general"
                confidence = 0.0
                strategy = RouteStrategy.FALLBACK

        # ── Stage 3: scope resolution ─────────────────────────────────────
        cls_cfg = self._config.task_classes.get(task_class)
        if cls_cfg and cls_cfg.scope_override:
            effective_scope = cls_cfg.scope_override
        elif has_cloud_directive(request.messages):
            effective_scope = "remote"
        else:
            effective_scope = self._config.provider_scope

        # ── Stage 4: model selection ──────────────────────────────────────
        profile = self._config.profiles.get(self._config.active_profile, {})
        profile_pref = profile.get("model_preference", "balanced")

        total_tokens = sum(
            len(m.get("content", "") if isinstance(m.get("content"), str) else "")
            for m in request.messages
        ) // 4

        backend_name, model_id, prefer_loaded = select_model(
            task_class=task_class,
            config=self._config,
            backends=self._backends,
            effective_scope=effective_scope,
            profile_pref=profile_pref,
            complexity=complexity_score(request.messages),
            estimated_tokens=total_tokens,
        )

        return RouteDecision(
            backend=backend_name,
            model_id=model_id,
            task_class=task_class,
            strategy=strategy,
            confidence=confidence,
            prefer_loaded=prefer_loaded,
            embedding=embedding,
        )

    def _extract_query(self, request: InferRequest) -> str:
        for m in reversed(request.messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, str):
                    return content
                if isinstance(content, list):
                    return " ".join(
                        p.get("text", "") for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    )
        return ""
