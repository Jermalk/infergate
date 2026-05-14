"""
OpenAI-compatible backend — covers OVH, ov_server REST, any OAI-compat API.
"""
import httpx

from infergate.types import InferRequest


class OpenAICompatBackend:
    def __init__(
        self,
        name: str,
        base_url: str,
        api_key: str = "",
        known_models: list[str] | None = None,
        is_local: bool = False,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._known_models = known_models
        self._is_local = is_local
        self._exclude_ids = exclude_ids
        self._fetched_models: list[str] = []

    @classmethod
    async def create(
        cls,
        name: str,
        base_url: str,
        api_key: str = "",
        is_local: bool = False,
        exclude_ids: frozenset[str] = frozenset(),
    ) -> "OpenAICompatBackend":
        """Async factory: creates backend and fetches model list immediately."""
        backend = cls(
            name=name,
            base_url=base_url,
            api_key=api_key,
            is_local=is_local,
            exclude_ids=exclude_ids,
        )
        await backend.fetch_models()
        return backend

    @property
    def is_local(self) -> bool:
        return self._is_local

    @property
    def routing_only(self) -> bool:
        return False

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        if self._known_models is not None:
            return [m for m in self._known_models if m not in self._exclude_ids]
        return [m for m in self._fetched_models if m not in self._exclude_ids]

    def loaded_model_ids(self) -> list[str]:
        return []

    async def fetch_models(self) -> list[str]:
        """Populate model list from GET /v1/models."""
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base_url}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        self._fetched_models = [m["id"] for m in data.get("data", [])]
        return self.available_models()

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        payload: dict = {
            "model":    model_id,
            "messages": request.messages,
            "stream":   request.stream,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = request.tools

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
