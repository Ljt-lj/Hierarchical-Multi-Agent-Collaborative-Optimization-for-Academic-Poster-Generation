"""文档树、内容树、海报树数据模型."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from poster_agent.models.visuals import VisualSpec


@dataclass
class RawNode:
    title: str
    content: str
    level: int = 1
    images: list[str] = field(default_factory=list)
    children: list[RawNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content,
            "level": self.level,
            "images": self.images,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RawNode:
        return cls(
            title=data.get("title", ""),
            content=data.get("content", ""),
            level=data.get("level", 1),
            images=data.get("images", []),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class ContentNode:
    title: str
    summary: str
    bullets: list[str] = field(default_factory=list)
    weight: float = 0.1
    logic_links: list[str] = field(default_factory=list)
    visuals: list[VisualSpec] = field(default_factory=list)
    image_paths: list[str] = field(default_factory=list)
    children: list[ContentNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "bullets": self.bullets,
            "weight": self.weight,
            "logic_links": self.logic_links,
            "visuals": [v.to_dict() for v in self.visuals],
            "image_paths": self.image_paths,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentNode:
        links = data.get("logic_links", [])
        normalized_links = []
        for link in links:
            if isinstance(link, str):
                normalized_links.append(link)
            elif isinstance(link, dict):
                normalized_links.append(str(link.get("target", link.get("relation", link))))
            else:
                normalized_links.append(str(link))
        weight_raw = data.get("weight")
        weight = float(weight_raw) if weight_raw is not None else 0.1
        bullets = data.get("bullets") or []
        visuals_raw = data.get("visuals") or []
        return cls(
            title=data.get("title", ""),
            summary=data.get("summary", "") or "",
            bullets=bullets,
            weight=weight,
            logic_links=normalized_links,
            visuals=[VisualSpec.from_dict(v) if isinstance(v, dict) else v for v in visuals_raw],
            image_paths=data.get("image_paths") or [],
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class LayoutRect:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class PosterNode:
    title: str
    summary: str
    bullets: list[str]
    rect: LayoutRect
    bg_color: tuple[int, int, int] = (245, 247, 250)
    text_color: tuple[int, int, int] = (30, 30, 30)
    font_size_title: int = 36
    font_size_body: int = 22
    image_path: str | None = None
    image_paths: list[str] = field(default_factory=list)
    visual_path: str | None = None
    visual_type: str | None = None
    layout_mode: str = "text_only"
    full_width: bool = False
    accent_color: tuple[int, int, int] = (52, 99, 170)
    section_icon: str | None = None
    is_header: bool = False
    children: list[PosterNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "bullets": self.bullets,
            "rect": {
                "x": self.rect.x,
                "y": self.rect.y,
                "width": self.rect.width,
                "height": self.rect.height,
            },
            "bg_color": self.bg_color,
            "text_color": self.text_color,
            "font_size_title": self.font_size_title,
            "font_size_body": self.font_size_body,
            "image_path": self.image_path,
            "image_paths": self.image_paths,
            "visual_path": self.visual_path,
            "visual_type": self.visual_type,
            "layout_mode": self.layout_mode,
            "full_width": self.full_width,
            "accent_color": self.accent_color,
            "section_icon": self.section_icon,
            "is_header": self.is_header,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PosterNode:
        r = data.get("rect") or {}
        rect = LayoutRect(
            int(r.get("x", 0)),
            int(r.get("y", 0)),
            int(r.get("width", 100)),
            int(r.get("height", 100)),
        )

        def tup(key: str, default: tuple[int, int, int]) -> tuple[int, int, int]:
            v = data.get(key, default)
            return tuple(v) if isinstance(v, (list, tuple)) else default  # type: ignore[return-value]

        return cls(
            title=data.get("title", ""),
            summary=data.get("summary", ""),
            bullets=list(data.get("bullets") or []),
            rect=rect,
            bg_color=tup("bg_color", (245, 247, 250)),
            text_color=tup("text_color", (30, 30, 30)),
            font_size_title=int(data.get("font_size_title", 36)),
            font_size_body=int(data.get("font_size_body", 22)),
            image_path=data.get("image_path"),
            image_paths=list(data.get("image_paths") or []),
            visual_path=data.get("visual_path"),
            visual_type=data.get("visual_type"),
            layout_mode=data.get("layout_mode", "text_only"),
            full_width=bool(data.get("full_width", False)),
            accent_color=tup("accent_color", (52, 99, 170)),
            section_icon=data.get("section_icon"),
            is_header=bool(data.get("is_header", False)),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )


@dataclass
class EvaluationScore:
    semantic_completeness: float
    layout_balance: float
    image_text_match: float
    iteration_count: int = 0
    feedback: str = ""

    @property
    def overall(self) -> float:
        return (
            0.4 * self.semantic_completeness
            + 0.35 * self.layout_balance
            + 0.25 * self.image_text_match
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_completeness": self.semantic_completeness,
            "layout_balance": self.layout_balance,
            "image_text_match": self.image_text_match,
            "overall": self.overall,
            "iteration_count": self.iteration_count,
            "feedback": self.feedback,
        }
