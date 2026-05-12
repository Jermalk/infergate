from typing import Protocol
from typing import runtime_checkable

from infergate.types import InferRequest


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provides text embeddings for semantic routing."""

    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Backend(Protocol):
    """Represents one inference backend."""

    @property
    def is_local(self) -> bool: ...

    def name(self) -> str: ...
    def available_models(self) -> list[str]: ...
    def loaded_model_ids(self) -> list[str]: ...

    async def chat(self, request: InferRequest, model_id: str) -> dict: ...
