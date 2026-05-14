"""
Fast-path signal detection — all O(1) or O(n_keywords). No embedding, always < 1 ms.
"""
import re

from infergate.config import RouterSettings
from infergate.types import InferRequest


_CLOUD_DIRECTIVE_RE = re.compile(r'#(ovh|cloud)\b', re.IGNORECASE)
# Default pattern used when task_class_directive is called without a compiled pattern.
# Deployments that add custom task classes should pass a Router-built pattern instead.
_DEFAULT_TASK_DIRECTIVE_RE = re.compile(r'#(code|document|general)\b', re.IGNORECASE)


def text_content(msg: dict) -> str:
    """Extract plain text from a message, handling both string and multimodal content."""
    content = msg.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def last_user_text(messages: list[dict]) -> str:
    """Return the text of the most recent user message, or empty string."""
    return next(
        (text_content(m) for m in reversed(messages) if m.get("role") == "user"),
        "",
    )


def has_images(messages: list[dict]) -> bool:
    """True if any message contains an image_url content part."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            if any(
                isinstance(p, dict) and p.get("type") == "image_url"
                for p in content
            ):
                return True
    return False


def task_class_directive(
    messages: list[dict],
    pattern: re.Pattern | None = None,
) -> str | None:
    """Return task_class if the last user message contains a #<task_class> tag.

    pattern should be a compiled regex built from the live task class names (built
    once at Router init). When None, falls back to the hardcoded default set
    {code, document, general} for backward compatibility.
    Only the last user message is checked.
    """
    p = pattern if pattern is not None else _DEFAULT_TASK_DIRECTIVE_RE
    m = p.search(last_user_text(messages))
    return m.group(1).lower() if m else None


def has_cloud_directive(messages: list[dict]) -> bool:
    """True if the last user message contains #ovh or #cloud."""
    return bool(_CLOUD_DIRECTIVE_RE.search(last_user_text(messages)))


def detect_signal(
    req: InferRequest,
    settings: RouterSettings,
    *,
    images_present: bool | None = None,
) -> str | None:
    """Return task_class if a non-directive fast-path signal fires, else None.

    Covers signals that carry objective evidence about request type. Hashtag
    directives (#code, #document, #general) are handled separately by the
    Router so it can assign RouteStrategy.KEYWORD rather than SIGNAL.

    Priority order:
      1. image content  → "vision"
      2. client tools   → settings.tools_task_class
      3. long context   → "document"
      4. keyword match  → task_class from settings.keywords

    images_present may be passed by the caller to avoid a redundant has_images() scan.
    Token count excludes system messages; uses char/4 heuristic.
    """
    if (images_present if images_present is not None else has_images(req.messages)):
        return "vision"

    if req.tools:
        return settings.tools_task_class

    total_tokens = sum(
        len(text_content(m)) for m in req.messages if m.get("role") != "system"
    ) // 4
    if total_tokens > settings.long_context_tokens:
        return "document"

    last_text = last_user_text(req.messages)
    if last_text:
        text_lower = last_text.lower()
        for task_class, keywords in settings.keywords.items():
            if any(kw.lower() in text_lower for kw in keywords):
                return task_class

    return None
