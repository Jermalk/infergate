from dataclasses import dataclass
from dataclasses import field


@dataclass
class ModelDescriptor:
    id:        str
    backend:   str           # must match a registered Backend.name()
    tier:      str           # "fast" | "balanced" | "best"
    ctx_limit: int = 32768


@dataclass
class TaskClassConfig:
    description:    str
    models:         list[ModelDescriptor] = field(default_factory=list)
    examples:       list[str]            = field(default_factory=list)
    scope_override: str | None = None    # "local" | "remote" | "hybrid"


@dataclass
class RouterSettings:
    embedding_min_confidence: float = 0.72
    long_context_tokens:      int   = 4000
    keywords:                 dict[str, list[str]] = field(default_factory=dict)


@dataclass
class RouterConfig:
    task_classes:   dict[str, TaskClassConfig]
    router:         RouterSettings = field(default_factory=RouterSettings)
    provider_scope: str = "local"    # "local" | "remote" | "hybrid"
    active_profile: str = "fast"
    profiles:       dict[str, dict] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "RouterConfig":
        router_raw = data.get("router", {})
        router_settings = RouterSettings(
            embedding_min_confidence=router_raw.get(
                "embedding_min_confidence",
                router_raw.get("embedding_threshold", 0.72),  # ov_server compat key
            ),
            long_context_tokens=router_raw.get("long_context_tokens", 4000),
            keywords=router_raw.get("keywords", {}),
        )

        task_classes: dict[str, TaskClassConfig] = {}
        for name, tc_raw in data.get("task_classes", {}).items():
            models = [
                ModelDescriptor(
                    id=m["id"],
                    backend=m.get("backend", m.get("provider", "")),
                    tier=m.get("tier", "fast"),
                    ctx_limit=m.get("ctx_limit", m.get("max_context_tokens", 32768)),
                )
                for m in tc_raw.get("models", [])
            ]
            task_classes[name] = TaskClassConfig(
                description=tc_raw.get("description", ""),
                models=models,
                examples=tc_raw.get("examples", []),
                scope_override=tc_raw.get("scope_override"),
            )

        return cls(
            task_classes=task_classes,
            router=router_settings,
            provider_scope=data.get("provider_scope", "local"),
            active_profile=data.get("active_profile", "fast"),
            profiles=data.get("profiles", {}),
        )
