# InferGate

Semantic routing library for AI inference backends. Routes each request to the
best available model based on what the request is asking for — not on
hardcoded rules or manual model selection.

Built as a standalone Python library with no GPU or OpenVINO dependency.
Extracted and generalised from production routing logic in
[ov_server](https://github.com/Jermalk/ov_server).

---

## The problem it solves

Running multiple LLM backends — a fast local model, a capable local VLM, a
cloud API — means every request needs a routing decision:

- Is this a vision task? Send to the VLM.
- Is this a 10 000-token document? The 4 k-context local model cannot handle it.
- Is this a quick factual question? The smallest fast model is enough.
- Is this a complex code review? Escalate to the best available model.

Doing this in application code means routing logic scattered across every
integration point, with no consistency and no observability. InferGate
centralises this into a single `router.decide(request)` call.

---

## How it works

Every request passes through a four-stage pipeline:

```
┌─ Stage 1: directive check ──────────────────────────────────────────────┐
│  #code / #document / #general tag in last user message?                 │
│  → assign task class immediately, strategy = KEYWORD                    │
└─────────────────────────────────────────────────────────────────────────┘
         │ no directive
┌─ Stage 2: signal detection ─────────────────────────────────────────────┐
│  O(1) checks in priority order:                                         │
│    image content  → "vision"      (image_url in message content)        │
│    client tools   → "web_search"  (tools list present)                  │
│    long context   → "document"    (char/4 estimate > long_context_tokens)│
│    keyword match  → task class    (configurable phrase list)            │
│  → assign task class, strategy = SIGNAL                                 │
└─────────────────────────────────────────────────────────────────────────┘
         │ no signal fired
┌─ Stage 3: embedding classification ────────────────────────────────────┐
│  Embed last user message with sentence-transformers                     │
│  Compute cosine similarity against per-class centroids                  │
│  score ≥ threshold  → best match,  strategy = EMBEDDING                 │
│  score < threshold  → "general",   strategy = FALLBACK                  │
└─────────────────────────────────────────────────────────────────────────┘
         │ task class known
┌─ Stage 4: scope resolution ─────────────────────────────────────────────┐
│  per-class scope_override  >  #ovh/#cloud directive  >  global scope    │
└─────────────────────────────────────────────────────────────────────────┘
         │ eligible backends determined
┌─ Stage 5: model selection ──────────────────────────────────────────────┐
│  1. Prefer loaded (warm VRAM) fast-tier models                          │
│  2. Pick by tier: fastest → balanced → best                             │
│  3. High complexity (> 0.65) promotes balanced → best                   │
│  4. Skip models where ctx_limit < estimated prompt tokens               │
└─────────────────────────────────────────────────────────────────────────┘
         ↓
    RouteDecision(backend, model_id, task_class, strategy, confidence, …)
```

Stages 1 and 2 are O(1) — no embedding, no network, always under 1 ms.
Stage 3 runs only when no signal matches, and runs in `run_in_executor` so
it never blocks the event loop.

---

## Installation

```bash
pip install infergate
```

For the demo gateway:

```bash
pip install "infergate[demo]"
```

For development:

```bash
git clone https://github.com/Jermalk/infergate
cd infergate
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,demo]"
```

Requires Python 3.10+.

---

## Quick start — routing only

```python
import asyncio
from infergate import Router, RouterConfig, InferRequest
from infergate.backends.ollama import OllamaBackend
from infergate.embeddings import SentenceTransformerProvider

async def main():
    backend = await OllamaBackend.create()          # discovers models from /api/tags
    config  = RouterConfig.from_dict({
        "provider_scope": "local",
        "task_classes": {
            "code":    {"description": "Code generation and debugging",
                        "models": [{"id": "qwen2.5-coder:14b", "backend": "ollama", "tier": "fast"}]},
            "general": {"description": "General questions and conversation",
                        "models": [{"id": "qwen2.5:3b", "backend": "ollama", "tier": "fast"}]},
        },
    })

    provider = SentenceTransformerProvider()
    router   = Router(config, backends={"ollama": backend}, embedding_provider=provider)
    await router.load_embeddings()                  # must be called before decide()

    decision = await router.decide(InferRequest(
        messages=[{"role": "user", "content": "Write a binary search in Python."}]
    ))
    print(decision.backend, decision.model_id, decision.strategy)
    # → ollama  qwen2.5-coder:14b  RouteStrategy.EMBEDDING

asyncio.run(main())
```

`Router.decide()` returns a `RouteDecision` and does **not** execute the request.
The caller chooses what to do with the decision — call `backend.chat()`, forward
to another service, log it, or anything else.

---

## Demo gateway

The `demo/` directory contains a ready-to-run FastAPI gateway that wires
infergate routing with full request execution across Ollama, ov_server, and
OVH AI Endpoints backends.

### Prerequisites

| Component | Notes |
|---|---|
| Ollama | Running at `localhost:11434` |
| ov_server | Optional. OAI-compat instance at any URL. |
| OVH AI Endpoints | Optional. Requires API key. |
| sentence-transformers | Loaded on first startup (≈ 600 MB cached). |

### Run

```bash
export INFERGATE_OVH_API_KEY=<your-key>
export INFERGATE_OV_SERVER_URL=http://localhost:11435/v1   # optional

cd demo
./run_demo.sh
```

Or without the script:

```bash
INFERGATE_OVH_API_KEY=<key> uvicorn gateway:app --app-dir demo --port 8080
```

### Environment variables

All settings use the `INFERGATE_` prefix and can also be placed in a `.env`
file in the working directory.

| Variable | Default | Description |
|---|---|---|
| `INFERGATE_CONFIG` | `demo/config.yaml` | Path to routing config file |
| `INFERGATE_OLLAMA_URL` | `http://localhost:11434` | Ollama base URL |
| `INFERGATE_OV_SERVER_URL` | *(empty — disabled)* | ov_server base URL including `/v1` |
| `INFERGATE_OVH_API_KEY` | *(empty — disabled)* | OVH bearer token |
| `INFERGATE_OVH_BASE_URL` | `https://oai.endpoints.kepler.ai.cloud.ovh.net/v1` | OVH endpoint |
| `INFERGATE_EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` | Sentence-transformers model |
| `INFERGATE_LOG_LEVEL` | `INFO` | Logging level |

### Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/v1/chat/completions` | OAI-compatible chat endpoint with routing |
| `GET` | `/v1/models` | Aggregated model list across all connected backends |
| `GET` | `/health` | Backend status and model counts |

Every response from `/v1/chat/completions` includes routing metadata headers:

```
X-InferGate-Backend:   ollama
X-InferGate-Model:     qwen2.5-coder:14b
X-InferGate-TaskClass: code
X-InferGate-Strategy:  keyword
```

### Demo requests

```bash
# Run the bundled demo script against a running gateway
INFERGATE_GW=http://localhost:8080 ./demo/demo_requests.sh
```

The script exercises three routing paths:

1. `#code` hashtag directive → strategy `keyword`, routes to fast-tier coder model
2. Plain factual question → strategy `embedding`, classified by cosine similarity
3. `implement …` phrase → strategy `signal`, keyword match fires before embedding

---

## Configuration reference

The routing config is a YAML (or JSON) file consumed by `RouterConfig.from_dict()`.

```yaml
provider_scope: hybrid          # "local" | "remote" | "hybrid" — default scope

active_profile: fast
profiles:
  fast:
    model_preference: fastest   # "fastest" | "balanced" | "best"
  best:
    model_preference: best

router:
  embedding_min_confidence: 0.72  # cosine threshold; below → fall back to "general"
  long_context_tokens: 4000       # token estimate (char/4) triggering document class
  keywords:                       # keyword → task class signal
    code:
      - "write a function"
      - "implement"

task_classes:
  code:
    description: "Code generation, debugging, and software engineering tasks."
    examples:                     # optional; improves centroid quality
      - "Write a Python function to parse JSON."
      - "Fix the null pointer exception in this code."
    scope_override: hybrid        # overrides provider_scope for this class
    models:
      - id: qwen2.5-coder:14b
        backend: ollama           # must match Backend.name()
        tier: fast                # "fast" | "balanced" | "best"
        ctx_limit: 32768          # max prompt tokens; model is skipped if exceeded
```

### Scope system

Controls which backends are eligible for a request. Three levels of override:

| Priority | Source | Example |
|---|---|---|
| Highest | `scope_override` in task class | `scope_override: remote` |
| Middle | `#ovh` or `#cloud` tag in message | forces `remote` |
| Lowest | Global `provider_scope` | `provider_scope: hybrid` |

| Scope | Eligible backends |
|---|---|
| `local` | Backends with `is_local=True` (Ollama, ov_server) |
| `remote` | Backends with `is_local=False` (OVH, any cloud API) |
| `hybrid` | All registered backends |

### User-facing directives

These tags in the **last user message** are intercepted by the router before
any other logic:

| Tag | Effect |
|---|---|
| `#code` | Force task class → `code`, strategy = `keyword` |
| `#document` | Force task class → `document`, strategy = `keyword` |
| `#general` | Force task class → `general`, strategy = `keyword` |
| `#ovh` | Force scope → `remote` (does not change task class) |
| `#cloud` | Same as `#ovh` |

Tags are case-insensitive and can appear anywhere in the message.

---

## Backends

### OllamaBackend

```python
from infergate.backends.ollama import OllamaBackend

backend = await OllamaBackend.create(
    base_url="http://localhost:11434",
    name="ollama",
)
```

- Discovers available models from `GET /api/tags`.
- Tracks models currently loaded in memory via `GET /api/ps` — used by
  the "prefer loaded" selection logic to prefer warm VRAM.
- `is_local = True` always.

### OpenAICompatBackend

```python
from infergate.backends.openai_compat import OpenAICompatBackend

backend = await OpenAICompatBackend.create(
    name="ovh",
    base_url="https://oai.endpoints.kepler.ai.cloud.ovh.net/v1",
    api_key="...",
    is_local=False,
    exclude_ids=frozenset({"Auto"}),   # filter sentinel / meta model IDs
)
```

Covers any OpenAI-compatible API: OVH AI Endpoints, ov_server, vLLM, LM Studio,
Together AI, Fireworks, etc. The `base_url` must include the versioned path
(e.g. `/v1`) — the backend appends `/models` and `/chat/completions` directly.

- `known_models`: supply a static list instead of fetching from `/models`.
- `exclude_ids`: frozenset of model IDs to hide from routing (e.g. routing
  sentinels or non-chat models like embedding endpoints).
- `loaded_model_ids()` returns `[]` — remote APIs have no concept of "loaded".

### Custom backends

Implement the `Backend` protocol:

```python
from infergate.protocols import Backend
from infergate.types import InferRequest

class MyBackend:
    @property
    def is_local(self) -> bool:
        return True

    def name(self) -> str:
        return "my_backend"

    def available_models(self) -> list[str]:
        return ["my-model"]

    def loaded_model_ids(self) -> list[str]:
        return []                            # return [] for remote / unknown

    async def chat(self, request: InferRequest, model_id: str) -> dict:
        # return an OpenAI-compatible response dict
        ...
```

No base class or registration required. Any object satisfying the protocol
can be passed to `Router(backends={"my_backend": MyBackend()})`.

---

## RouteDecision fields

```python
@dataclass
class RouteDecision:
    backend:       str             # Backend.name() of the selected backend
    model_id:      str             # model ID on that backend
    task_class:    str             # "code" / "document" / "general" / …
    strategy:      RouteStrategy   # KEYWORD | SIGNAL | EMBEDDING | FALLBACK
    confidence:    float           # cosine similarity score; 1.0 for signal routes
    prefer_loaded: bool            # True when a warm VRAM model was preferred
    embedding:     list[float] | None  # request embedding vector, if computed
```

`confidence` and `strategy` are useful for monitoring: low confidence with
`FALLBACK` strategy means the embedding model was uncertain and fell back to
the general class. `prefer_loaded` tells you whether VRAM state influenced
the pick.

---

## Custom EmbeddingProvider

```python
from infergate.protocols import EmbeddingProvider
import asyncio

class MyEmbeddingProvider:
    async def embed(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_one, text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._encode_batch, texts)

    def _encode_one(self, text: str) -> list[float]: ...
    def _encode_batch(self, texts: list[str]) -> list[list[float]]: ...
```

CPU-bound work **must** run in `run_in_executor`. The default
`SentenceTransformerProvider` uses `intfloat/multilingual-e5-large` — a
multilingual model that handles non-English queries without any configuration.

---

## Kubernetes deployment

The demo gateway follows the 12-factor config pattern. Use it as a reference
for production deployments:

```yaml
# Deployment env section
env:
  - name: INFERGATE_OVH_API_KEY
    valueFrom:
      secretKeyRef:
        name: infergate-secrets
        key: ovh-api-key
  - name: INFERGATE_CONFIG
    value: /etc/infergate/config.yaml

# Mount routing config from a ConfigMap
volumeMounts:
  - name: routing-config
    mountPath: /etc/infergate

volumes:
  - name: routing-config
    configMap:
      name: infergate-routing-config
```

The routing topology (task classes, model lists, thresholds) lives in the
ConfigMap. Secrets (API keys) are injected as environment variables.
Neither requires a pod restart to take effect via a rolling update.

---

## Development

```bash
# Run tests — no GPU, no network, no real model required
pytest

# Type check
mypy src/infergate --ignore-missing-imports

# Format
black src tests demo
```

All tests use a hash-based `MockEmbeddingProvider` that produces deterministic
unit vectors without loading any model. The full test suite runs in under 1
second.

---

## Design decisions

**No GPU dependency in the library.** infergate is a routing and dispatch
layer. Model inference stays in the backends. The only CPU-bound work is the
embedding forward pass for task classification, which runs in `run_in_executor`.

**Protocol-based backends, not a plugin registry.** Any class satisfying the
`Backend` structural protocol works without registration, inheritance, or
decorators. Python's `isinstance(obj, Backend)` check validates at runtime.

**Signal detection before embedding.** For requests with obvious signals —
an image, a tools list, a very long context — running an embedding model is
waste. The O(1) signal check handles the majority of clearly-typed requests.
Embedding classification only runs when the request is genuinely ambiguous.

**Scope system over provider names.** Scope values (`local`, `remote`,
`hybrid`) are backend-agnostic. A backend is local or remote by its
`is_local` flag, not by its name. This avoids coupling routing config to
specific provider names.

**config.yaml for topology, env vars for secrets.** Routing structure
(task classes, models, thresholds) belongs in version control. Credentials
and infrastructure URLs belong in the environment. The `pydantic-settings`
`BaseSettings` class enforces this split in the demo gateway.
