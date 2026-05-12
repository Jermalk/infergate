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
        self._loaded: list[str] = []

    @classmethod
    async def create(
        cls,
        base_url: str = "http://localhost:11434",
        name: str = "ollama",
    ) -> "OllamaBackend":
        """Async factory: creates backend and populates model lists immediately."""
        backend = cls(base_url=base_url, name=name)
        await backend.fetch_models()
        return backend

    @property
    def is_local(self) -> bool:
        return True

    def name(self) -> str:
        return self._name

    def available_models(self) -> list[str]:
        return list(self._models)

    def loaded_model_ids(self) -> list[str]:
        """Models currently running in Ollama memory (from /api/ps cache)."""
        return list(self._loaded)

    async def fetch_models(self) -> list[str]:
        """Populate available and loaded model lists from Ollama.

        Calls /api/tags for all pulled models and /api/ps for models
        currently loaded in memory. Call this before first use, or to refresh.
        """
        async with httpx.AsyncClient(timeout=10) as client:
            tags_resp = await client.get(f"{self._base_url}/api/tags")
            tags_resp.raise_for_status()
            self._models = [m["name"] for m in tags_resp.json().get("models", [])]

            try:
                ps_resp = await client.get(f"{self._base_url}/api/ps")
                ps_resp.raise_for_status()
                self._loaded = [m["name"] for m in ps_resp.json().get("models", [])]
            except Exception:
                self._loaded = []

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

        prompt_tokens = raw.get("prompt_eval_count", 0)
        completion_tokens = raw.get("eval_count", 0)

        return {
            "id":      raw.get("model", model_id),
            "object":  "chat.completion",
            "choices": [{
                "index":         0,
                "message":       raw.get("message", {}),
                "finish_reason": raw.get("done_reason", "stop"),
            }],
            "usage": {
                "prompt_tokens":     prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens":      prompt_tokens + completion_tokens,
            },
        }
