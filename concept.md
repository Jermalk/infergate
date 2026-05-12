# InferGate — Package Concept

Extraction of intelligent routing logic from `ov_server`.
Delivers semantic routing for any AI inference backend (LLM, VLM, STT, TTS, embeddings).
Uses bridge plugins to connect to one or more backends and routes each request to the
best available model based on request semantics — not hardcoded rules.

---

## Package name: `infergate`

PyPI namespace already registered. Import root: `infergate`.

---

## Session protocol

Follow the same CLAUDE.md session protocol used in ov_server:
PROGRESS.md (NOW section only on re-entry), SESSION.md (crash recovery),
SCRATCHPAD.md (cleared each session), DECISIONS.md (append-only).
Create these files at project root on first session.

---

## Source files for extraction

Extract from the ov_server project. Key source files:

| Source file | What to extract |
|---|---|
| `router.py` | Signal detection, `_route_by_embedding()`, `complexity_score()`, `_select_model()`, directive parsing |
| `server_config.py` | RouterConfig shape — task_classes, router settings, profiles, scope |
| `model_manager.py` | EmbeddingProvider pattern — tokenizer + model forward pass + mean pooling + L2 normalise |
| `tests/test_pure.py` | Migrate all pure routing tests to infergate test suite |

---

## Implementation order

1. Extract routing logic from ov_server into `infergate` package.
2. Reconnect ov_server to use `infergate` — this proves the abstraction is correct.
   ov_server becomes the production proof that infergate works. Reconnecting forces
   every abstraction leak to surface before the API surface is committed.
3. Build bridge-set demo: thin FastAPI gateway using infergate with Ollama + OVH backends.

---

## Package structure

```
infergate/
├── pyproject.toml
├── README.md
├── LICENSE                      ← MIT
├── src/
│   └── infergate/
│       ├── __init__.py          ← public API exports
│       ├── config.py            ← RouterConfig, TaskClassConfig, RouterSettings dataclasses
│       ├── types.py             ← InferRequest, RouteDecision, RouteStrategy
│       ├── protocols.py         ← Backend and EmbeddingProvider Protocol classes
│       ├── signals.py           ← has_images, long_context check, keyword directives
│       ├── embeddings.py        ← centroid computation, cosine similarity routing
│       ├── selector.py          ← _select_model, complexity_score, tier escalation
│       └── router.py            ← Router class — public entry point
├── infergate/backends/
│   ├── __init__.py
│   ├── openai_compat.py         ← OpenAICompatBackend (OVH, ov_server, any OAI-compat API)
│   └── ollama.py                ← OllamaBackend with /api/tags model discovery
└── tests/
    ├── conftest.py
    └── test_routing.py          ← migrated from ov_server/tests/test_pure.py + new cases
```

---

## Public API (`__init__.py`)

```python
from infergate.router import Router
from infergate.types import RouteDecision, RouteStrategy, InferRequest
from infergate.config import RouterConfig
from infergate.protocols import Backend, EmbeddingProvider

__version__ = "0.1.0"
__all__ = [
    "Router",
    "RouteDecision", "RouteStrategy", "InferRequest",
    "RouterConfig",
    "Backend", "EmbeddingProvider",
]
```

---

## Core types (`types.py`)

```python
from dataclasses import dataclass, field
from enum import Enum

class RouteStrategy(str, Enum):
    SIGNAL    = "signal"             # fast path: image/tools/long_context detected
    KEYWORD   = "keyword"            # #code #document #general directive
    EMBEDDING = "embedding"          # cosine similarity above threshold
    FALLBACK  = "embedding_fallback" # cosine similarity below threshold → general

@dataclass
class InferRequest:
    """Minimal request shape needed by the router. Does NOT proxy — caller handles execution."""
    messages:   list[dict]        # [{"role": "user", "content": "..."}]
    model:      str | None = None # optional caller hint; router may override
    max_tokens: int | None = None # used in complexity scoring
    stream:     bool = False

@dataclass
class RouteDecision:
    backend:       str            # registered backend name e.g. "loc", "ovh"
    model_id:      str            # model identifier on that backend
    task_class:    str            # e.g. "code", "document", "general", "vision"
    strategy:      RouteStrategy
    confidence:    float          # cosine similarity score; 1.0 for signal-based routes
    prefer_loaded: bool = False   # True if selection was influenced by in-memory model
    embedding:     list[float] | None = None  # request embedding vector, if computed
```

---

## Protocols (`protocols.py`)

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbeddingProvider(Protocol):
    """Provides text embeddings for semantic routing."""
    async def embed(self, text: str) -> list[float]: ...
    async def embed_batch(self, texts: list[str]) -> list[list[float]]: ...

@runtime_checkable
class Backend(Protocol):
    """Represents one inference backend."""

    def name(self) -> str: ...
    # Unique identifier used in config and RouteDecision.backend

    def available_models(self) -> list[str]: ...
    # All model IDs this backend can serve

    def loaded_model_ids(self) -> list[str]: ...
    # Models currently warm in memory. Empty list for remote backends.

    async def chat(self, request: "InferRequest", model_id: str) -> dict: ...
    # Optional execution layer. Returns raw JSON-compatible response dict.
    # Not required for pure routing — callers that only need router.decide()
    # can ignore this method. Bridges implement this for the demo gateway.
```

**Two usage modes:**

- **Routing only:** `router.decide(request)` → `RouteDecision`. Caller handles
  execution however it wants. No dependency on bridge execution.
- **Routing + execution:** `router.decide(request)` then `backend.chat(request,
  decision.model_id)`. The demo gateway uses this mode.

---

## Config dataclasses (`config.py`)

```python
from dataclasses import dataclass, field

@dataclass
class ModelDescriptor:
    id:        str
    backend:   str           # must match a registered Backend.name()
    tier:      str           # "fast" | "balanced" | "best"
    ctx_limit: int = 32768   # max context tokens

@dataclass
class TaskClassConfig:
    description:    str                    # natural language — embedded at startup into centroid
    models:         list[ModelDescriptor] = field(default_factory=list)
    scope_override: str | None = None      # "local" | "remote" | "hybrid"

@dataclass
class RouterSettings:
    embedding_min_confidence: float = 0.72  # cosine threshold; below → fall back to general
    long_context_tokens:      int   = 4000  # token count triggering document class

@dataclass
class RouterConfig:
    task_classes:   dict[str, TaskClassConfig]
    router:         RouterSettings = field(default_factory=RouterSettings)
    provider_scope: str = "local"   # "local" | "hybrid" | "remote"
    active_profile: str = "fast"
    profiles:       dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig": ...
    # Parses from config.json / config.yaml dict. Mirrors ov_server config shape.
```

---

## Router class (`router.py`)

```python
class Router:
    def __init__(self, config: RouterConfig, backends: dict[str, Backend],
                 embedding_provider: EmbeddingProvider | None = None): ...

    @classmethod
    def from_config(cls, config: dict | RouterConfig,
                    backends: dict[str, Backend],
                    embedding_provider: EmbeddingProvider | None = None) -> "Router": ...

    async def load_embeddings(self) -> None:
        """Embed all task class descriptions and store as centroids.
        Must be awaited before first decide() call.
        Runs in run_in_executor — does not block event loop."""
        ...

    async def decide(self, request: InferRequest) -> RouteDecision:
        """Main entry point. Returns routing decision without executing the request.

        Pipeline:
        1. Signal detection  — O(1), no embedding needed
        2. Embedding routing — CPU-bound, run_in_executor
        3. Scope filter      — eliminates ineligible backends
        4. Model selection   — prefer loaded → tier → complexity → context check
        """
        ...
```

---

## Scope system

Controls which backends are eligible for each request. Central to the hybrid routing
value proposition. Must be preserved exactly from ov_server.

| Scope | Eligible backends |
|---|---|
| `local` | Backends with `is_local=True` |
| `remote` | Backends with `is_local=False` |
| `hybrid` | All registered backends |

Resolution order (highest priority first):
1. Per-task-class `scope_override` in config
2. `#ovh` or `#cloud` keyword directive in request (forces remote)
3. Global `RouterConfig.provider_scope`

---

## Embedding routing — precise behaviour

Extracted from `ov_server/router.py:_route_by_embedding()`. Preserve exactly:

1. Truncate query to 2048 chars before tokenisation
2. Tokenise with `padding=True, truncation=True, max_length=512`
3. Forward pass through embedding model
4. Mean-pool `last_hidden_state` across sequence dimension
5. L2-normalise the vector: `vec / max(norm, 1e-9)`
6. Compute `np.dot(vec, centroid)` for each task class centroid
7. If `best_score >= 0.72` → return `(best_class, best_score)` — confident
8. If `best_score < 0.72` → return `("general", best_score)` — fallback

**Critical:** the embedding forward pass is CPU-bound and MUST run in
`asyncio.get_event_loop().run_in_executor(None, ...)` — never call synchronously
from an async handler.

**Default EmbeddingProvider:** `sentence-transformers` with
`intfloat/multilingual-e5-large`. Accept any `EmbeddingProvider` via constructor
injection. No OpenVINO dependency anywhere in the library.

---

## Bridge implementations

### `OpenAICompatBackend` — covers OVH, ov_server REST, any OpenAI-compatible API

```python
class OpenAICompatBackend:
    def __init__(self, name: str, base_url: str, api_key: str = "",
                 known_models: list[str] | None = None,
                 is_local: bool = False):
        # known_models: static list OR None → fetched from GET /v1/models at init
        ...
```

### `OllamaBackend` — with model discovery

```python
class OllamaBackend:
    def __init__(self, base_url: str = "http://localhost:11434", name: str = "ollama"):
        # Discovers models via GET /api/tags
        # is_local = True always
        ...
```

### `OVLocalBackend` (lives in ov_server, NOT in infergate)

```python
# ov_server/backends/local.py — the only backend touching OV internals
class OVLocalBackend:
    def name(self) -> str: return "loc"
    def available_models(self) -> list[str]:
        return list(AVAILABLE_MODELS) + list(AVAILABLE_VLM_MODELS)
    def loaded_model_ids(self) -> list[str]:
        return list(model_manager.loaded_models)
    async def chat(self, request, model_id):
        # delegates to existing ov_server local inference path — nothing changes
        ...
```

---

## ov_server reconnection — what router.py becomes

```python
# ov_server/router.py — after reconnection (~40 lines, down from ~280)
from infergate import Router, RouteDecision
from infergate.backends import OpenAICompatBackend
from .backends.local import OVLocalBackend

_router: Router | None = None

async def init_router(cfg: dict) -> None:
    global _router
    backends: dict = {"loc": OVLocalBackend()}
    ovh = cfg.get("providers", {}).get("ovh", {})
    if ovh:
        backends["ovh"] = OpenAICompatBackend(
            name="ovh",
            base_url=ovh["base_url"],
            api_key=os.environ.get(ovh["api_key_env"], ""),
            is_local=False,
        )
    _router = Router.from_config(cfg, backends=backends)
    await _router.load_embeddings()

async def decide(req) -> RouteDecision:
    from infergate.types import InferRequest
    return await _router.decide(InferRequest(
        messages=[m.model_dump() for m in req.messages],
        model=req.model,
        max_tokens=req.max_tokens,
        stream=req.stream,
    ))
```

---

## Demo gateway (Step 3)

Minimal FastAPI app (`demo/gateway.py`) demonstrating hybrid routing without Arc GPU
or OpenVINO. Runs on any machine with Docker and an OVH API key.

```
demo/
├── gateway.py          ← thin FastAPI: POST /v1/chat/completions
├── config.yaml         ← task_classes with Ollama + OVH backends
├── docker-compose.yml  ← ollama + infergate-demo services
├── Dockerfile
├── run_demo.sh         ← sends 3 requests, prints routing decisions
└── README.md           ← 5-step setup
```

`demo/config.yaml`:
```yaml
provider_scope: hybrid

backends:
  local:
    type: ollama
    base_url: http://ollama:11434
    is_local: true
  ovh:
    type: openai_compat
    base_url: https://oai.endpoints.kepler.ai.cloud.ovh.net/v1
    api_key_env: OVH_API_KEY
    is_local: false

router:
  embedding_min_confidence: 0.72
  long_context_tokens: 4000

task_classes:
  code:
    description: "Write, review, debug, refactor, or explain code and programming concepts"
    models:
      - id: qwen3:8b   backend: local  tier: fast
      - id: Qwen3-32B  backend: ovh    tier: best
  document:
    description: "Summarise, analyse, or extract information from long documents"
    models:
      - id: qwen3:8b   backend: local  tier: fast
      - id: Qwen3-32B  backend: ovh    tier: best
  general:
    description: "General conversation, questions, and explanations"
    models:
      - id: qwen3:8b   backend: local  tier: fast
      - id: Qwen3-32B  backend: ovh    tier: balanced
```

Three demo requests showing three routing outcomes:
1. `"what is 2+2"` → general, fast tier → local Ollama
2. `"refactor this Python function [200 lines]"` → code, embedding → OVH
3. `"summarise this document [8000 words]"` → document, long_context signal → OVH

Response headers must expose routing decisions:
`X-InferGate-Backend`, `X-InferGate-Model`, `X-InferGate-TaskClass`, `X-InferGate-Strategy`

---

## pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "infergate"
version = "0.1.0"
description = "Semantic routing layer for AI inference backends"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10"
authors = [{ name = "Jerzy", email = "your@email.com" }]
keywords = ["ai", "llm", "routing", "inference", "gateway", "semantic"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Topic :: Software Development :: Libraries",
    "Programming Language :: Python :: 3",
]
dependencies = [
    "numpy>=1.24",
    "httpx>=0.27",
    "sentence-transformers>=3.0",
    "pydantic>=2.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0", "pytest-asyncio>=0.23", "black", "mypy"]

[project.urls]
Homepage = "https://github.com/yourusername/infergate"
```

---

## Tests requirement

- All tests pass without a GPU, without OpenVINO, without network access.
- Mock EmbeddingProvider in tests — do not call a real embedding model.
- Migrate relevant tests from `ov_server/tests/test_pure.py`.
- Minimum coverage: signal detection, cosine threshold decision, scope filtering,
  complexity scoring, prefer-loaded logic, context overflow escalation.
- Use `pytest-asyncio` for async tests.

---

## Coding standards

- Python 3.10+ typing: `X | None`, `list[X]`, `dict[K, V]` — no legacy `typing` imports.
- All embedding forward passes in `run_in_executor`.
- All `async def` functions calling CPU-bound code use `asyncio.timeout()`.
- `black` and `mypy --ignore-missing-imports` pass clean.
- No module-level mutable globals — encapsulate state in the `Router` instance.
- One import per line.

---

## Routing flow diagram

The complete routing pipeline. The slow path (embedding classification) is only reached
when no signal is detected — this is where the cosine similarity threshold controls
whether the request gets a confident task class or falls back to `general`.

<div class="mermaid">
flowchart TD
    REQ(["incoming request"])
    REQ --> SD
    SD{"signal check: \n has_image \n has_tools \n tokens more than 4k \n keyword/hashword present"}
    SD -->|"task class \n determined"| SCF
    SD -->|"no signal \n matched"| EP
    INIT[/"startup - init \n embed task descriptions \n store as centroids"/]
    subgraph SLOW["slow path - embedding classification"]
        direction TB
        EP["embed request text \n run_in_executor"]
        CS["cosine similarity \n request vector  each centroid"]
        TH{"best score >= 0.72?"}
        BM["task = best match \n strategy = embedding \n confidence = score"]
        FB["task = general \n strategy = fallback \n confidence = low_score"]
        EP --> CS
        CS --> TH
        TH -->|yes| BM
        TH -->|no| FB
    end
    INIT -. centroids .-> CS
    BM --> SCF
    FB --> SCF
    SCF["scope filter: \n local - hybrid - remote"]
    SCF --> LM
    LM{"loaded model \n in eligible pool?"}
    LM -->|"yes - prefer \n warm VRAM"| SL["select loaded model \n skip tier logic"]
    LM -->|no| PT["profile tier \n + complexity score \n fast - precise - laborious"]
    SL --> CTX
    PT --> CTX
    CTX{"context \n limit ok?"}
    CTX -->|yes| OUT
    CTX -->|"no - escalate \n to remote"| ES["larger context \n remote model"]
    ES --> OUT

    OUT(["RouteDecision \n backend - model_id - task_class \n strategy - confidence"])
</div>
