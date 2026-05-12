from dataclasses import dataclass
from dataclasses import field
from enum import Enum


class RouteStrategy(str, Enum):
    SIGNAL   = "signal"
    KEYWORD  = "keyword"
    EMBEDDING = "embedding"
    FALLBACK = "embedding_fallback"


@dataclass
class InferRequest:
    """Minimal request shape needed by the router. Does NOT proxy — caller handles execution."""
    messages:   list[dict]
    model:      str | None = None
    max_tokens: int | None = None
    stream:     bool = False
    tools:      list[dict] | None = None


@dataclass
class RouteDecision:
    backend:       str
    model_id:      str
    task_class:    str
    strategy:      RouteStrategy
    confidence:    float
    prefer_loaded: bool = False
    embedding:     list[float] | None = None
