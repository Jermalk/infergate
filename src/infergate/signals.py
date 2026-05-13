"""
Fast-path signal detection — all O(1) or O(n_keywords). No embedding, always < 1 ms.
"""
import re

from infergate.config import RouterSettings
from infergate.types import InferRequest


_CLOUD_DIRECTIVE_RE = re.compile(r'#(ovh|cloud)\b', re.IGNORECASE)
_TASK_DIRECTIVE_RE  = re.compile(r'#(code|document|general)\b', re.IGNORECASE)

# Classes that are only reachable via signal detection, never via embedding routing.
# compute_centroids() skips these so they don't pollute the similarity space.
_SIGNAL_ONLY_CLASSES: frozenset[str] = frozenset({"vision", "web_search"})


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


def task_class_directive(messages: list[dict]) -> str | None:
    """Return task_class if the last user message contains a #code, #document, or #general tag.

    Only the last user message is checked; earlier turns are ignored so the
    directive applies to the current request, not a historical one.
    """
    m = _TASK_DIRECTIVE_RE.search(last_user_text(messages))
    return m.group(1).lower() if m else None


def has_cloud_directive(messages: list[dict]) -> bool:
    """True if the last user message contains #ovh or #cloud."""
    return bool(_CLOUD_DIRECTIVE_RE.search(last_user_text(messages)))


def detect_signal(req: InferRequest, settings: RouterSettings) -> str | None:
    """Return task_class if a non-directive fast-path signal fires, else None.

    Covers signals that carry objective evidence about request type. Hashtag
    directives (#code, #document, #general) are handled separately by the
    Router so it can assign RouteStrategy.KEYWORD rather than SIGNAL.

    Priority order:
      1. image content  → "vision"
      2. client tools   → "web_search"
      3. long context   → "document"
      4. keyword match  → task_class from settings.keywords

    Token count for long-context detection excludes system messages: they are
    usually fixed infrastructure text and should not inflate the request size.
    Token estimate uses the char / 4 heuristic (good enough for routing).
    """
    if has_images(req.messages):
        return "vision"

    if req.tools:
        return "web_search"

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
