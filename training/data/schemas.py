"""SFT 样本与任务类型定义."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class TaskType(str, Enum):
    REFINER = "refiner"
    VISUAL = "visual"
    COMMENTER = "commenter"
    ORCHESTRATOR = "orchestrator"


@dataclass
class ChatMessage:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class SFTSample:
    task: str
    messages: list[ChatMessage]
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "messages": [m.to_dict() for m in self.messages],
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SFTSample:
        msgs = [ChatMessage(**m) for m in data.get("messages", [])]
        return cls(task=data.get("task", "refiner"), messages=msgs, meta=data.get("meta", {}))


REFINER_SYSTEM = (
    "You are an academic poster content refiner. Compress document content for poster display. "
    "Output valid JSON: {title, summary, bullets, weight, logic_links, children}. "
    "Keep 2-4 bullets per section, summary under 120 chars, weights 0-1."
)

VISUAL_SYSTEM = (
    "You design ONE scientific poster visual per section. "
    "Output JSON: {type, title, section_title, data}. "
    "Types: stat_cards, bar_chart, line_chart, flow_diagram, architecture, bullet_cards, pie_chart."
)

COMMENTER_SYSTEM = (
    "You evaluate academic poster quality. "
    "Output JSON: {score, issues, suggestions} with score 0-1."
)
