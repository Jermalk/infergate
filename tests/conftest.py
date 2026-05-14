"""
Shared fixtures and helpers for infergate tests.
No GPU, no network, no real embedding model required.
"""
import numpy as np
import pytest

from infergate.config import ModelDescriptor
from infergate.config import RouterConfig
from infergate.config import RouterSettings
from infergate.config import TaskClassConfig
from infergate.protocols import Backend
from infergate.protocols import EmbeddingProvider


class MockEmbeddingProvider:
    """Deterministic mock: each unique text gets a fixed unit vector based on its hash."""

    def _vec(self, text: str) -> list[float]:
        dim = 4
        seed = hash(text) % (2 ** 31)
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(dim)
        norm = np.linalg.norm(v)
        return (v / max(norm, 1e-9)).tolist()

    async def embed(self, text: str) -> list[float]:
        return self._vec(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class MockBackend:
    def __init__(
        self,
        name: str,
        models: list[str],
        loaded: list[str] | None = None,
        is_local: bool = True,
    ) -> None:
        self._name = name
        self._models = models
        self._loaded = loaded or []
        self._is_local = is_local

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def routing_only(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        return list(self._models)

    def loaded_model_ids(self) -> list[str]:
        return list(self._loaded)

    async def chat(self, request, model_id: str) -> dict:
        return {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}


@pytest.fixture
def mock_provider() -> MockEmbeddingProvider:
    return MockEmbeddingProvider()


@pytest.fixture
def local_backend() -> MockBackend:
    return MockBackend(
        name="loc",
        models=["small-llm", "big-llm"],
        loaded=["small-llm"],
        is_local=True,
    )


@pytest.fixture
def remote_backend() -> MockBackend:
    return MockBackend(
        name="ovh",
        models=["cloud-llm"],
        loaded=[],
        is_local=False,
    )


@pytest.fixture
def basic_config() -> RouterConfig:
    return RouterConfig(
        task_classes={
            "code": TaskClassConfig(
                description="Write, debug, and review code",
                models=[
                    ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                    ModelDescriptor(id="big-llm",   backend="loc", tier="best"),
                    ModelDescriptor(id="cloud-llm", backend="ovh", tier="best"),
                ],
            ),
            "document": TaskClassConfig(
                description="Analyse and summarise long documents",
                models=[
                    ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                    ModelDescriptor(id="cloud-llm", backend="ovh", tier="best"),
                ],
            ),
            "general": TaskClassConfig(
                description="General conversation and questions",
                models=[
                    ModelDescriptor(id="small-llm", backend="loc", tier="fast"),
                    ModelDescriptor(id="big-llm",   backend="loc", tier="balanced"),
                    ModelDescriptor(id="cloud-llm", backend="ovh", tier="best"),
                ],
            ),
        },
        router=RouterSettings(
            embedding_min_confidence=0.72,
            long_context_tokens=100,
            keywords={"code": ["fix this", "debug"]},
        ),
        provider_scope="local",
        active_profile="fast",
        profiles={
            "fast":    {"model_preference": "fastest"},
            "precise": {"model_preference": "balanced"},
            "best":    {"model_preference": "best"},
        },
    )
