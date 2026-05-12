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
    """Represents one inference backend.

    Two usage modes:
    - Routing only: implement everything except chat(). Router.decide() never
      calls chat(); the caller executes against the chosen backend itself.
    - Routing + execution: implement chat() too. The demo gateway uses this mode.

    Contract for loaded_model_ids():
      Return model IDs currently warm in memory (e.g. loaded into GPU VRAM).
      Remote backends with no concept of "loaded" must return an empty list.
      The router uses this to prefer warm models under the "fastest" profile.
    """

    @property
    def is_local(self) -> bool: ...

    def name(self) -> str: ...
    def available_models(self) -> list[str]: ...
    def loaded_model_ids(self) -> list[str]: ...

    async def chat(self, request: InferRequest, model_id: str) -> dict: ...
