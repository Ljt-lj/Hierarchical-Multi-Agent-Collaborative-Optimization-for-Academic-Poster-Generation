"""可视化规格：图表、流程图、统计卡等."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VisualSpec:
    type: str  # bar_chart, line_chart, flow_diagram, stat_cards, figure, pie_chart
    title: str = ""
    section_title: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    image_path: str | None = None
    width_ratio: float = 0.42

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "title": self.title,
            "section_title": self.section_title,
            "data": self.data,
            "image_path": self.image_path,
            "width_ratio": self.width_ratio,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisualSpec:
        raw_data = data.get("data")
        if isinstance(raw_data, dict):
            spec_data = raw_data
        elif isinstance(raw_data, list):
            spec_data = {"items": raw_data}
        else:
            spec_data = {}
        return cls(
            type=data.get("type", "icon"),
            title=data.get("title", ""),
            section_title=data.get("section_title", ""),
            data=spec_data,
            image_path=data.get("image_path"),
            width_ratio=float(data.get("width_ratio", 0.42)),
        )
