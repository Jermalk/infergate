"""
infergate demo gateway — thin FastAPI app that routes requests via infergate.

Backends:
  ollama  — local Ollama at localhost:11434
  ovh     — OVH AI Endpoints (OAI-compat) at INFERGATE_OVH_BASE_URL

Environment variables:
  INFERGATE_OVH_API_KEY   — OVH bearer token (required for OVH backend)
  INFERGATE_OVH_BASE_URL  — override OVH endpoint URL
  INFERGATE_CONFIG        — path to config.yaml (default: same dir as this file)
"""
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import yaml
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi import Request
from fastapi.responses import JSONResponse

from infergate import InferRequest
from infergate import Router
from infergate import RouterConfig
from infergate.backends.ollama import OllamaBackend
from infergate.backends.openai_compat import OpenAICompatBackend
from infergate.embeddings import SentenceTransformerProvider

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("infergate.demo")

_SCRIPT_DIR = Path(__file__).parent

# ── globals populated at startup ──────────────────────────────────────────────
router: Router | None = None
backends: dict[str, OllamaBackend | OpenAICompatBackend] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global router, backends

    config_path = Path(os.environ.get("INFERGATE_CONFIG", _SCRIPT_DIR / "config.yaml"))
    log.info("loading config from %s", config_path)
    cfg_dict = yaml.safe_load(config_path.read_text())
    config = RouterConfig.from_dict(cfg_dict)

    # ── Ollama ────────────────────────────────────────────────────────────────
    try:
        ollama = await OllamaBackend.create(base_url="http://localhost:11434", name="ollama")
        backends["ollama"] = ollama
        log.info("[ollama] models: %s", ollama.available_models())
    except Exception as exc:
        log.warning("[ollama] unavailable (%s) — skipping", exc)

    # ── OVH AI Endpoints ──────────────────────────────────────────────────────
    ovh_key = os.environ.get("INFERGATE_OVH_API_KEY", "")
    ovh_url = os.environ.get(
        "INFERGATE_OVH_BASE_URL",
        "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    )
    if ovh_key:
        try:
            ovh = await OpenAICompatBackend.create(
                name="ovh",
                base_url=ovh_url,
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
        provider = SentenceTransformerProvider()
        log.info("[embeddings] SentenceTransformerProvider loaded")
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
