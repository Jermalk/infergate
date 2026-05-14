from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Literal


EliminationReason = Literal["no_backend", "scope", "unavailable", "ctx_limit", "modality"]


@dataclass
class EliminatedCandidate:
    """One model that was considered and rejected during select_model()."""

    model_id: str
    backend:  str
    reason:   EliminationReason


@dataclass
class RouteTrace:
    """Optional routing trace attached to RouteDecision when decide(trace=True) is used."""

    eliminated:   list[EliminatedCandidate] = field(default_factory=list)
    scope_source: str = ""          # "class_override" | "cloud_directive" | "global"
    embedding_ms: float | None = None  # wall time of embed() call; None on signal/keyword path
    cache_hit:    bool | None = None  # True/False on embedding path; None when path not taken


class NoModelAvailable(Exception):
    """Raised by select_model when no backend has a reachable model for the given scope."""

    def __init__(self, task_class: str, scope: str) -> None:
        super().__init__(f"No model available for task_class='{task_class}' scope='{scope}'")
        self.task_class = task_class
        self.scope = scope


class RouteStrategy(str, Enum):
    SIGNAL    = "signal"            # image / tools / long-context / keyword match
    KEYWORD   = "keyword"           # explicit #code / #document / #general directive
    EMBEDDING = "embedding"         # cosine similarity above min_confidence threshold
    FALLBACK  = "embedding_fallback"  # cosine similarity below threshold → general
    RESELECT  = "reselect"          # caller-driven re-run via Router.reselect()


@dataclass
class InferRequest:
    """Minimal request shape passed to the router. Does NOT proxy — caller handles execution."""

    messages:   list[dict]
    model:      str | None        = None  # optional caller hint; router may override
    max_tokens: int | None        = None
    stream:     bool              = False
    tools:      list[dict] | None = None  # presence triggers tools_task_class signal
    force_tier: str | None        = None  # programmatic tier override (e.g. admin/assessor); use message directives (#code etc.) for user-initiated routing


@dataclass
class RouteDecision:
    """Routing outcome returned by Router.decide(). Contains no side-effects."""

    backend:       str             # registered backend name matching Backend.name()
    model_id:      str             # model identifier on that backend
    task_class:    str             # e.g. "code", "document", "general", "vision"
    strategy:      RouteStrategy
    confidence:    float           # cosine similarity score; 1.0 for signal-based routes
    prefer_loaded:    bool = False    # True when selection was influenced by in-memory model
    embedding:        list[float] | None = None  # request embedding vector, if computed
    task_directive:   str | None = None  # matched task-class directive (e.g. "code"), or None
    estimated_tokens: int = 0           # prompt token estimate (sum of text lengths // 4)
    trace:            RouteTrace | None = None  # populated only when decide(trace=True)
