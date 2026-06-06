"""P2PInstruct 任务分类（download / build_sft 共用）."""

from __future__ import annotations

REFINER_MIN_CHARS = 900
VISUAL_MAX_CHARS = 700
REFINER_CONTENT_MIN_CHARS = 1500  # P2P 论文：章节生成 ~3300 tokens，图注 ~192 tokens

VISUAL_USER_MARKERS = (
    "figure descriptor",
    "describe the figure",
    "describe this figure",
    "describe the image",
    "visual element",
    "caption",
    "chart",
    "table",
    "figure caption",
    "image caption",
)

REFINER_USER_MARKERS = (
    "section generator",
    "content generator",
    "poster section",
    "poster content",
    "compress",
    "summarize",
    "content tree",
    "refine",
    "bullet point",
    "abstract",
    "introduction",
    "conclusion",
    "methodology",
    "experiment",
    "document tree",
    "html generator",
)


def messages_to_pair(messages: list[dict]) -> tuple[str, str, str]:
    system = ""
    user_parts: list[str] = []
    assistant = ""
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system":
            system = content
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant = content
    return system, "\n\n".join(user_parts), assistant


def is_figure_description_task(user: str, assistant: str) -> bool:
    u, a = user.lower(), assistant.lower()
    if any(m in u for m in VISUAL_USER_MARKERS):
        return True
    if any(m in u for m in ("figure", "image", "visual", "caption")) and len(assistant) <= VISUAL_MAX_CHARS * 3:
        return True
    opening = a[:280]
    if opening.startswith(("this figure", "the figure", "this image", "the image")):
        return True
    if "illustrates" in opening and ("figure" in opening or "image" in opening):
        return True
    return False


def is_refiner_content(user: str, assistant: str) -> bool:
    """是否为章节/内容生成样本（非 figure descriptor）."""
    if not assistant.strip():
        return False
    if is_figure_description_task(user, assistant):
        return False
    n = len(assistant)
    u, a = user.lower(), assistant.lower()
    if any(m in a for m in ('"children"', '"bullets"', "## abstract", "## introduction")):
        return True
    if any(m in u for m in REFINER_USER_MARKERS):
        return True
    if n >= REFINER_CONTENT_MIN_CHARS:
        return True
    if n >= REFINER_MIN_CHARS and ("##" in assistant or "###" in assistant):
        return True
    return False


def is_visual_content(user: str, assistant: str) -> bool:
    return is_figure_description_task(user, assistant) or (
        len(assistant) <= VISUAL_MAX_CHARS * 2 and any(m in user.lower() for m in ("figure", "image", "visual"))
    )
