"""
infergate demo gateway — thin FastAPI app that routes requests via infergate.

Configuration is driven by pydantic-settings. All operational parameters come
from environment variables (prefixed INFERGATE_) or an optional .env file.
The routing topology (task classes, model lists) comes from the YAML config file.

Key environment variables:
  INFERGATE_OVH_API_KEY      OVH bearer token (required for OVH backend)
  INFERGATE_OVH_BASE_URL     override OVH endpoint URL
  INFERGATE_OLLAMA_URL       override Ollama base URL
  INFERGATE_CONFIG           path to config.yaml
  INFERGATE_EMBEDDING_MODEL  sentence-transformers model name
  INFERGATE_LOG_LEVEL        logging level (default INFO)

Kubernetes pattern:
  - ConfigMap  → mount config.yaml; set INFERGATE_CONFIG to mount path
  - Secret     → set INFERGATE_OVH_API_KEY from secretKeyRef
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict

from infergate import InferRequest
from infergate import Router
from infergate import RouterConfig
from infergate.backends.ollama import OllamaBackend
from infergate.backends.openai_compat import OpenAICompatBackend
from infergate.embeddings import SentenceTransformerProvider
from infergate.protocols import Backend

_SCRIPT_DIR = Path(__file__).parent


class Settings(BaseSettings):
    """Operational settings — sourced from env vars or .env file.

    All fields map to INFERGATE_<FIELD_NAME> environment variables.
    Routing topology (task classes, thresholds) stays in config.yaml.
    """

    model_config = SettingsConfigDict(
        env_prefix="INFERGATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    config: Path = _SCRIPT_DIR / "config.yaml"

    ovh_api_key: SecretStr = SecretStr("")
    ovh_base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"

    ollama_url: str = "http://localhost:11434"

    embedding_model: str = "intfloat/multilingual-e5-large"

    log_level: str = "INFO"


settings = Settings()

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    force=True,  # override any root-logger config already applied by the host process
)
log = logging.getLogger("infergate.demo")

# ── globals populated at startup ──────────────────────────────────────────────
router: Router | None = None
backends: dict[str, Backend] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, backends

    log.info("loading config from %s", settings.config)
    cfg_dict = yaml.safe_load(settings.config.read_text())
    config = RouterConfig.from_dict(cfg_dict)

    # ── Ollama ────────────────────────────────────────────────────────────────
    try:
        ollama = await OllamaBackend.create(base_url=settings.ollama_url, name="ollama")
        backends["ollama"] = ollama
        log.info("[ollama] models: %s", ollama.available_models())
    except Exception as exc:
        log.warning("[ollama] unavailable (%s) — skipping", exc)

    # ── OVH AI Endpoints ──────────────────────────────────────────────────────
    ovh_key = settings.ovh_api_key.get_secret_value()
    if ovh_key:
        try:
            ovh = await OpenAICompatBackend.create(
                name="ovh",
                base_url=settings.ovh_base_url,
                api_key=ovh_key,
                is_local=False,
            )
            backends["ovh"] = ovh
            log.info("[ovh] models (first 5): %s", ovh.available_models()[:5])
        except Exception as exc:
            log.warning("[ovh] unavailable (%s) — skipping", exc)
    else:
        log.warning("[ovh] INFERGATE_OVH_API_KEY not set — OVH backend disabled")

    if not backends:
        raise RuntimeError("no backends available — check Ollama and OVH credentials")

    # ── Router ────────────────────────────────────────────────────────────────
    try:
        provider = SentenceTransformerProvider(model_name=settings.embedding_model)
        log.info("[embeddings] provider loaded: %s", settings.embedding_model)
    except Exception as exc:
        log.warning("[embeddings] failed to load model (%s) — embedding routing disabled", exc)
        provider = None

    router = Router(config=config, backends=backends, embedding_provider=provider)
    await router.load_embeddings()
    log.info("[router] ready — backends: %s", list(backends))

    yield

    log.info("gateway shutdown")


app = FastAPI(title="infergate demo", lifespan=lifespan)


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> JSONResponse:
    body = await request.json()

    messages = body.get("messages")
    if not messages:
        raise HTTPException(status_code=400, detail="messages required")

    infer_req = InferRequest(
        messages=messages,
        model=body.get("model"),
        max_tokens=body.get("max_tokens"),
        stream=body.get("stream", False),
        tools=body.get("tools"),
    )

    assert router is not None, "router not initialised — lifespan startup failed"
    decision = await router.decide(infer_req)
    log.info(
        "[route] task=%s strategy=%s backend=%s model=%s confidence=%.2f",
        decision.task_class,
        decision.strategy.value,
        decision.backend,
        decision.model_id,
        decision.confidence,
    )

    backend = backends.get(decision.backend)
    if backend is None:
        raise HTTPException(
            status_code=503,
            detail=f"backend '{decision.backend}' not available",
        )

    try:
        result = await backend.chat(infer_req, decision.model_id)
    except Exception as exc:
        log.error("[chat] backend=%s model=%s error: %s", decision.backend, decision.model_id, exc)
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    headers = {
        "X-InferGate-Backend":   decision.backend,
        "X-InferGate-Model":     decision.model_id,
        "X-InferGate-TaskClass": decision.task_class,
        "X-InferGate-Strategy":  decision.strategy.value,
    }
    return JSONResponse(content=result, headers=headers)


@app.get("/health")
async def health() -> dict:
    return {
        "status":   "ok",
        "backends": {
            name: backend.available_models()
            for name, backend in backends.items()
        },
    }


@app.get("/v1/models")
async def list_models() -> dict:
    models = []
    for backend_name, backend in backends.items():
        for model_id in backend.available_models():
            models.append({
                "id":       model_id,
                "object":   "model",
                "backend":  backend_name,
                "is_local": backend.is_local,
            })
    return {"object": "list", "data": models}
