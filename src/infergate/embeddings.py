"""
Centroid computation and embedding-based routing.
All EmbeddingProvider calls are async; CPU-bound work lives inside the provider.
"""
import asyncio

import numpy as np

from infergate.config import TaskClassConfig
from infergate.protocols import EmbeddingProvider
from infergate.signals import _SIGNAL_ONLY_CLASSES


async def compute_centroids(
    task_classes: dict[str, TaskClassConfig],
    provider: EmbeddingProvider,
) -> dict[str, np.ndarray]:
    """Embed task class descriptions+examples and store L2-normalised centroids.

    Skips signal-only classes (vision, web_search) — those never reach embedding routing.
    """
    centroids: dict[str, np.ndarray] = {}
    for name, cls_cfg in task_classes.items():
        if name in _SIGNAL_ONLY_CLASSES:
            continue
        texts: list[str] = []
        if cls_cfg.description:
            texts.append(cls_cfg.description)
        texts.extend(cls_cfg.examples)
        if not texts:
            continue
        vecs = await provider.embed_batch(texts)
        arr = np.array(vecs, dtype=float)
        centroid = arr.mean(axis=0)
        norm = np.linalg.norm(centroid)
        centroids[name] = centroid / max(norm, 1e-9)
    return centroids


async def route_by_embedding(
    query: str,
    centroids: dict[str, np.ndarray],
    provider: EmbeddingProvider,
    min_confidence: float = 0.72,
) -> tuple[str, float, list[float] | None]:
    """Return (task_class, cosine_similarity, embedding_vector).

    Returns ("general", 0.0, None) when centroids are unavailable.
    Falls back to "general" when best score < min_confidence.
    """
    if not centroids:
        return ("general", 0.0, None)

    vec_list = await provider.embed(query[:2048])
    vec = np.array(vec_list, dtype=float)
    norm = np.linalg.norm(vec)
    vec = vec / max(norm, 1e-9)

    best_class, best_score = "general", 0.0
    for task_class, centroid in centroids.items():
        score = float(np.dot(vec, centroid))
        if score > best_score:
            best_class, best_score = task_class, score

    if best_score < min_confidence:
        best_class = "general"

    return (best_class, best_score, vec.tolist())


class SentenceTransformerProvider:
    """Default EmbeddingProvider backed by sentence-transformers.

    CPU-bound inference runs in run_in_executor so the event loop stays free.
    """

    def __init__(self, model_name: str = "intfloat/multilingual-e5-large") -> None:
        self._model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)

    def _encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        vecs = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vecs]

    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, self._encode, [text])
        return results[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode, texts)
