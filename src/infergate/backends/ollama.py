"""
Ollama backend — always local, discovers models via GET /api/tags.
"""
import httpx

from infergate.types import InferRequest


class OllamaBackend:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        name: str = "ollama",
    ) -> None:
        self._name = name
        self._base_url = base_url.rstrip("/")
        self._models: list[str] = []

    @property
    def is_local(self) -> bool:
        return True

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        return list(self._models)

    def loaded_model_ids(self) -> list[str]:
        return list(self._models)  # Ollama keeps all pulled models immediately available

    async def fetch_models(self) -> list[str]:
        """Discover models via GET /api/tags."""
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"{self._base_url}/api/tags")
            resp.raise_for_status()
            data = resp.json()
        self._models = [m["name"] for m in data.get("models", [])]
        return self._models

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        payload: dict = {
            "model":    model_id,
            "messages": request.messages,
            "stream":   False,
        }
        if request.max_tokens is not None:
            payload["options"] = {"num_predict": request.max_tokens}

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            raw = resp.json()

        # Normalise to OpenAI-compat shape
        return {
            "id":      raw.get("model", model_id),
            "object":  "chat.completion",
            "choices": [{
                "index":         0,
                "message":       raw.get("message", {}),
                "finish_reason": "stop",
            }],
        }
