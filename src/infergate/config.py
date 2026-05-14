"""
Configuration dataclasses for infergate.

RouterConfig is the single source of truth for routing behaviour. It is
intentionally backend-agnostic: backend instances are registered at runtime,
not named here beyond a string key that must match Backend.name().
"""
from dataclasses import dataclass
from dataclasses import field
from typing import Literal


Scope    = Literal["local", "remote", "hybrid"]
Tier     = Literal["fast", "balanced", "best"]
Modality = Literal["text", "vision", "any"]


@dataclass
class ModelDescriptor:
    """One model entry in a task class, bound to a named backend."""

    id:        str
    backend:   str      # must match a registered Backend.name()
    tier:      Tier     # controls preference order within select_model
    ctx_limit: int      = 32768
    modality:  Modality = "text"  # "vision" for VLMs; "any" for multimodal-capable models


@dataclass
class TaskClassConfig:
    """Routing behaviour for one semantic task class."""

    description:    str
    models:         list[ModelDescriptor] = field(default_factory=list)
    examples:       list[str]             = field(default_factory=list)
    scope_override: Scope | None = None   # overrides global provider_scope for this class
    signal_only:    bool         = False  # True → skip centroid; class reached only via signal


@dataclass
class RouterSettings:
    """Thresholds and keyword tables that tune the routing pipeline."""

    embedding_min_confidence: float = 0.72  # cosine threshold; below → fall back to general
    long_context_tokens:      int   = 4000  # token count triggering the document class
    keywords:        dict[str, list[str]] = field(default_factory=dict)  # class → trigger phrases
    tools_task_class: str = "web_search"   # task class assigned when req.tools is non-empty
    complexity_promote_fast_threshold: float | None = None  # None = disabled; set to promote fastest → balanced


@dataclass
class RouterConfig:
    """Complete routing configuration. Construct directly or via from_dict()."""

    task_classes:   dict[str, TaskClassConfig]
    router:         RouterSettings = field(default_factory=RouterSettings)
    provider_scope: Scope = "local"   # default scope when no override applies
    active_profile: str   = "fast"
    profiles:       dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        """Parse from a config.yaml / config.json dict.

        Accepts both current infergate field names and legacy ov_server names:
          embedding_threshold       → embedding_min_confidence
          provider / backend        → backend (in model descriptors)
          max_context_tokens        → ctx_limit
        """
        router_raw = data.get("router", {})
        router_settings = RouterSettings(
            embedding_min_confidence=router_raw.get(
                "embedding_min_confidence",
                router_raw.get("embedding_threshold", 0.72),
            ),
            long_context_tokens=router_raw.get("long_context_tokens", 4000),
            keywords=router_raw.get("keywords", {}),
            tools_task_class=router_raw.get("tools_task_class", "web_search"),
            complexity_promote_fast_threshold=router_raw.get("complexity_promote_fast_threshold"),
        )

        task_classes: dict[str, TaskClassConfig] = {}
        for name, tc_raw in data.get("task_classes", {}).items():
            models = [
                ModelDescriptor(
                    id=m["id"],
                    backend=m.get("backend", m.get("provider", "")),
                    tier=m.get("tier", "fast"),
                    ctx_limit=m.get("ctx_limit", m.get("max_context_tokens", 32768)),
                    modality=m.get("modality", "text"),
                )
                for m in tc_raw.get("models", [])
            ]
            task_classes[name] = TaskClassConfig(
                description=tc_raw.get("description", ""),
                models=models,
                examples=tc_raw.get("examples", []),
                scope_override=tc_raw.get("scope_override"),
                signal_only=tc_raw.get("signal_only", False),
            )

        return cls(
            task_classes=task_classes,
            router=router_settings,
            provider_scope=data.get("provider_scope", "local"),
            active_profile=data.get("active_profile", "fast"),
            profiles=data.get("profiles", {}),
        )
