"""布局智能体：学术海报网格布局（摘要通栏 + 双栏 masonry）."""

from __future__ import annotations

from poster_agent.config import PosterConfig
from poster_agent.models.trees import ContentNode, PosterNode
from poster_agent.render.grid_layout import GridLayoutEngine


class LayoutAgent:
    def __init__(self, config: PosterConfig):
        self.config = config
        self.paper_title = "Academic Poster"
        self._engine = GridLayoutEngine(config)

    def layout(
        self,
        content: ContentNode,
        height_boost: dict[str, float] | None = None,
    ) -> PosterNode:
        self.paper_title = content.title or self.paper_title
        return self._engine.layout(content, height_boost=height_boost)
