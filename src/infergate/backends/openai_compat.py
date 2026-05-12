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
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._known_models = known_models  # None → fetched lazily from /v1/models
        self._is_local = is_local
        self._fetched_models: list[str] = []

    @property
    def is_local(self) -> bool:
        return self._is_local

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        if self._known_models is not None:
            return list(self._known_models)
        return list(self._fetched_models)

    def loaded_model_ids(self) -> list[str]:
        return []

    async def fetch_models(self) -> list[str]:
        """Populate model list from GET /v1/models. Called at init when known_models is None."""
        headers = {"Authorization": f"Bearer {self._api_key}"} if self._api_key else {}
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base_url}/v1/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        self._fetched_models = [m["id"] for m in data.get("data", [])]
        return self._fetched_models

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
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
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            resp.raise_for_status()
            return resp.json()
