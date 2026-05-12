"""
Fast-path signal detection — all O(1) or O(n_keywords). No embedding, always < 1 ms.
"""
import re

from infergate.config import RouterSettings
from infergate.types import InferRequest


_CLOUD_DIRECTIVE_RE = re.compile(r'#(ovh|cloud)\b', re.IGNORECASE)
_TASK_DIRECTIVE_RE  = re.compile(r'#(code|document|general)\b', re.IGNORECASE)

_TASK_DIRECTIVE_MAP: dict[str, str] = {
    "code":     "code",
    "document": "document",
    "general":  "general",
}

_SIGNAL_ONLY_CLASSES: frozenset[str] = frozenset({"vision", "web_search"})


def text_content(msg: dict) -> str:
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def has_images(messages: list[dict]) -> bool:
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            if any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            ):
                return True
    return False


def task_class_directive(messages: list[dict]) -> str | None:
    """Return task_class if the last user message contains #code, #document, or #general."""
    last_user = next(
        (text_content(m) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    m = _TASK_DIRECTIVE_RE.search(last_user)
    return _TASK_DIRECTIVE_MAP.get(m.group(1).lower()) if m else None


def has_cloud_directive(messages: list[dict]) -> bool:
    """True if the last user message contains #ovh or #cloud."""
    last_user = next(
        (text_content(m) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )
    return bool(_CLOUD_DIRECTIVE_RE.search(last_user))


def detect_signal(req: InferRequest, settings: RouterSettings) -> str | None:
    """Return task_class if a fast-path signal fires, else None.

    Priority order:
      0. #code / #document / #general hashtag directive
      1. image content  → "vision"
      2. client tools   → "web_search"
      3. long context   → "document"
      4. keyword match  → task_class from settings.keywords
    """
    directive = task_class_directive(req.messages)
    if directive:
        return directive

    if has_images(req.messages):
        return "vision"

    if req.tools:
        return "web_search"

    total_tokens = sum(
        len(text_content(m)) for m in req.messages if m.get("role") != "system"
    ) // 4
    if total_tokens > settings.long_context_tokens:
        return "document"

    last_user_text = next(
        (text_content(m) for m in reversed(req.messages) if m.get("role") == "user"),
        "",
    )
    if last_user_text:
        text_lower = last_user_text.lower()
        for task_class, keywords in settings.keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return task_class

    return None
