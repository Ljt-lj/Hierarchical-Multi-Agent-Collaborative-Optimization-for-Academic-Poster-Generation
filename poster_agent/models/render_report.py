"""渲染阶段溢出/密度报告（Paper2Poster Painter–Commenter 反馈）."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SectionRenderReport:
    title: str
    bullets_total: int
    bullets_shown: int
    has_visual: bool
    has_paper_figure: bool
    body_font: int
    area: int
    char_count: int

    @property
    def overflow(self) -> bool:
        return self.bullets_shown < self.bullets_total

    @property
    def sparse(self) -> bool:
        if self.area <= 0:
            return True
        density = self.char_count / max(self.area / 800, 1)
        return density < 0.12 and not self.overflow

    @property
    def missing_visual(self) -> bool:
        t = self.title.lower()
        if any(k in t for k in ("reference", "ref", "参考")):
            return False
        return not self.has_visual and not self.has_paper_figure


@dataclass
class PosterRenderReport:
    sections: list[SectionRenderReport] = field(default_factory=list)

    @property
    def overflow_sections(self) -> list[str]:
        return [s.title for s in self.sections if s.overflow]

    @property
    def sparse_sections(self) -> list[str]:
        return [s.title for s in self.sections if s.sparse]

    @property
    def missing_visual_sections(self) -> list[str]:
        return [s.title for s in self.sections if s.missing_visual]

    def overflow_rate(self) -> float:
        if not self.sections:
            return 0.0
        return sum(1 for s in self.sections if s.overflow) / len(self.sections)
