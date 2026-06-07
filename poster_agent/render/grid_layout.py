"""学术海报网格布局：摘要通栏 + 多栏 masonry，无重叠、等间距."""

from __future__ import annotations

from dataclasses import dataclass

from poster_agent.config import PosterConfig
from poster_agent.models.trees import ContentNode, LayoutRect, PosterNode

# 区块之间统一间距（布局矩形边缘到边缘）
BLOCK_GAP = 16
HEADER_H = 280
FOOTER_H = 50
INNER_PAD = 20


@dataclass
class _SectionPlan:
    node: ContentNode
    layout_mode: str
    estimated_h: int
    body_font: int
    title_font: int
    bar_h: int
    full_width: bool = False


class GridLayoutEngine:
    def __init__(self, config: PosterConfig):
        self.config = config

    def layout(self, content: ContentNode, height_boost: dict[str, float] | None = None) -> PosterNode:
        sections = _collect_sections(content)
        if not sections:
            sections = [content]

        ordered = _order_sections(sections)
        abstract = _pop_abstract(ordered)

        content_x = self.config.margin
        content_y = HEADER_H + BLOCK_GAP
        content_w = self.config.width - 2 * self.config.margin
        content_bottom = self.config.height - FOOTER_H - BLOCK_GAP

        children: list[PosterNode] = []

        if abstract:
            plan = _plan_section(abstract, content_w, self.config, full_width=True)
            h = min(plan.estimated_h, int((content_bottom - content_y) * 0.28))
            h = max(h, plan.estimated_h // 2)
            rect = LayoutRect(content_x, content_y, content_w, h)
            children.append(_make_node(abstract, rect, plan))
            body_start_y = content_y + h + BLOCK_GAP
        else:
            body_start_y = content_y

        body_sections = ordered
        if not body_sections:
            return _wrap_root(content, children, content_x, content_y, content_w, content_bottom - content_y)

        num_cols = 3 if len(body_sections) >= 7 else 2
        col_w = (content_w - BLOCK_GAP * (num_cols - 1)) // num_cols
        available_h = content_bottom - body_start_y

        plans = [_plan_section(s, col_w, self.config) for s in body_sections]
        weights = [_section_layout_weight(s) for s in body_sections]
        total_est = sum(p.estimated_h * w for p, w in zip(plans, weights)) + BLOCK_GAP * (len(plans) // num_cols)
        scale = min(1.0, available_h / max(total_est, 1))

        col_heights = [body_start_y] * num_cols
        placed: list[tuple[PosterNode, int]] = []

        for plan, section, weight in zip(plans, body_sections, weights):
            boost = 1.0
            if height_boost:
                for key, mult in height_boost.items():
                    if key.lower() in section.title.lower() or section.title.lower() in key.lower():
                        boost = max(boost, mult)
            h = max(int(plan.estimated_h * weight * scale * boost), _min_section_height(plan))
            col = min(range(num_cols), key=lambda i: col_heights[i])
            x = content_x + col * (col_w + BLOCK_GAP)
            y = col_heights[col]

            max_h = content_bottom - y
            if h > max_h:
                h = max(max_h, _min_section_height(plan))

            rect = LayoutRect(x, y, col_w, h)
            node = _make_node(section, rect, plan)
            children.append(node)
            placed.append((node, col))
            col_heights[col] = y + h + BLOCK_GAP

        _resolve_overlaps(placed, content_bottom)
        _stretch_column_tails(placed, content_bottom, body_sections)

        return _wrap_root(content, children, content_x, content_y, content_w, content_bottom - content_y)


def _wrap_root(
    content: ContentNode,
    children: list[PosterNode],
    x: int,
    y: int,
    w: int,
    h: int,
) -> PosterNode:
    return PosterNode(
        title="Poster Root",
        summary="",
        bullets=[],
        rect=LayoutRect(x, y, w, h),
        bg_color=(248, 249, 252),
        children=children,
    )


def _make_node(section: ContentNode, rect: LayoutRect, plan: _SectionPlan) -> PosterNode:
    accent = _section_accent(section.title)
    visual = section.visuals[0] if section.visuals else None
    images = list(section.image_paths)
    if visual and visual.image_path and visual.image_path not in images:
        images.insert(0, visual.image_path)

    return PosterNode(
        title=section.title,
        summary=section.summary,
        bullets=section.bullets,
        rect=rect,
        bg_color=(255, 255, 255),
        accent_color=accent,
        section_icon=_section_icon(section.title),
        font_size_title=plan.title_font,
        font_size_body=plan.body_font,
        image_path=images[0] if images else None,
        image_paths=images,
        visual_path=visual.image_path if visual else None,
        visual_type=visual.type if visual else None,
        layout_mode=plan.layout_mode,
        full_width=plan.full_width,
    )


def _plan_fonts(node: ContentNode, layout: str, config: PosterConfig) -> tuple[int, int, int]:
    t = node.title.lower()
    if _is_reference(t):
        body, title = 28, 36
    elif len(node.bullets) >= 4:
        body, title = 32, 40
    else:
        body, title = 34, 42
    body = max(body, config.min_body_font if not _is_reference(t) else 28)
    title = max(title, config.min_title_font if not _is_reference(t) else 36)
    bar_h = title + 28
    return body, title, bar_h


def _plan_section(node: ContentNode, col_w: int, config: PosterConfig, full_width: bool = False) -> _SectionPlan:
    layout = _choose_layout(node)
    body_font, title_font, bar_h = _plan_fonts(node, layout, config)
    text_h = _text_height(node, col_w, layout, body_font)
    fig_h = _figure_height(node, col_w, layout)
    chrome = bar_h + INNER_PAD * 3
    if layout == "side_by_side":
        total = chrome + max(text_h, fig_h)
    else:
        total = chrome + text_h + fig_h
    return _SectionPlan(node, layout, int(total), body_font, title_font, bar_h, full_width)


def _min_section_height(plan: _SectionPlan) -> int:
    return plan.bar_h + INNER_PAD * 3 + 120


def _choose_layout(node: ContentNode) -> str:
    t = node.title.lower()
    has_fig = bool(node.image_paths or node.visuals)
    if not has_fig:
        return "text_dense" if _is_reference(t) else "text_only"
    if "conclusion" in t or "结论" in t:
        return "figure_top"
    if _is_result(t) or _is_experiment(t):
        return "figure_top"
    if _is_intro(t):
        return "figure_top"
    if _is_method(t):
        return "figure_bottom"
    return "figure_bottom"


def _text_height(node: ContentNode, col_w: int, layout: str, body_font: int) -> int:
    chars_per_line = max(col_w // max(body_font // 2 + 4, 10), 16)
    if layout == "side_by_side":
        chars_per_line = max(col_w // max(body_font + 8, 20), 14)
    elif layout == "figure_adaptive":
        chars_per_line = max(col_w // max(body_font // 2 + 4, 10), 16)
    lines = 0
    if node.summary:
        lines += max(1, len(node.summary) // chars_per_line + 1)
    for b in node.bullets:
        lines += max(1, len(b) // chars_per_line + 1)
    line_h = body_font + 10
    return lines * line_h + 12


def _figure_height(node: ContentNode, col_w: int, layout: str) -> int:
    if not (node.image_paths or node.visuals):
        return 0
    t = node.title.lower()
    if layout == "figure_top":
        n_figs = min(len(node.image_paths or []) + len(node.visuals or []), 2)
        ratio = 0.34 if "conclusion" in t or "结论" in t else 0.30
        if _is_intro(t):
            ratio = 0.24
        base = int(col_w * (ratio if n_figs >= 1 else 0.28))
        return base
    if layout == "side_by_side":
        return int(col_w * 0.42)
    if layout == "figure_adaptive":
        return int(col_w * 0.32)
    if layout == "figure_bottom":
        return int(col_w * 0.30)
    return 0


def _resolve_overlaps(placed: list[tuple[PosterNode, int]], content_bottom: int) -> None:
    """确保同列区块不重叠，保持 BLOCK_GAP."""
    by_col: dict[int, list[PosterNode]] = {}
    for node, col in placed:
        by_col.setdefault(col, []).append(node)

    for nodes in by_col.values():
        nodes.sort(key=lambda n: n.rect.y)
        for i in range(len(nodes) - 1):
            curr, nxt = nodes[i], nodes[i + 1]
            max_bottom = nxt.rect.y - BLOCK_GAP
            allowed_h = max_bottom - curr.rect.y
            if curr.rect.height > allowed_h:
                curr.rect = LayoutRect(curr.rect.x, curr.rect.y, curr.rect.width, max(allowed_h, _min_height(curr)))


def _stretch_column_tails(
    placed: list[tuple[PosterNode, int]],
    content_bottom: int,
    sections: list[ContentNode],
) -> None:
    """按区块权重分配列底剩余空间，结论/实验多分配，引言少分配."""
    by_col: dict[int, list[PosterNode]] = {}
    for node, col in placed:
        by_col.setdefault(col, []).append(node)

    title_map = {s.title: s for s in sections}

    for nodes in by_col.values():
        nodes.sort(key=lambda n: n.rect.y)
        last = nodes[-1]
        slack = content_bottom - (last.rect.y + last.rect.height)
        if slack <= BLOCK_GAP:
            continue
        weights = [_section_stretch_weight(title_map.get(n.title, None)) for n in nodes]
        total_w = sum(weights) or len(nodes)
        shift = 0
        for i, node in enumerate(nodes):
            extra = int(slack * weights[i] / total_w)
            node.rect = LayoutRect(
                node.rect.x,
                node.rect.y + shift,
                node.rect.width,
                node.rect.height + extra,
            )
            shift += extra


def _section_layout_weight(node: ContentNode) -> float:
    t = node.title.lower()
    heuristic = 1.0
    if _is_intro(t):
        heuristic = 0.78
    elif "conclusion" in t or "结论" in t:
        heuristic = 1.28
    elif _is_experiment(t):
        heuristic = 1.08
    elif _is_method(t):
        heuristic = 0.95
    elif "background" in t or "背景" in t:
        heuristic = 0.92
    content_w = float(node.weight) if node.weight and node.weight > 0 else 1.0
    return 0.55 * heuristic + 0.45 * content_w


def _section_stretch_weight(node: ContentNode | None) -> float:
    if node is None:
        return 1.0
    t = node.title.lower()
    if _is_intro(t):
        return 0.25
    if "conclusion" in t or "结论" in t:
        return 2.2
    if _is_experiment(t):
        return 1.15
    if _is_method(t):
        return 0.75
    if "background" in t or "背景" in t:
        return 0.85
    return 1.0


def _min_height(node: PosterNode) -> int:
    return node.font_size_title + 28 + INNER_PAD * 3 + 80


def _order_sections(sections: list[ContentNode]) -> list[ContentNode]:
    priority = {
        "abstract": 0, "摘要": 0,
        "introduction": 1, "intro": 1, "引言": 1, "背景": 1,
        "method": 2, "approach": 2, "方法": 2, "模型": 2,
        "experiment": 3, "result": 3, "实验": 3, "结果": 3, "讨论": 3,
        "conclusion": 4, "discussion": 4, "结论": 4,
        "reference": 5, "acknowledg": 5, "参考文献": 5, "致谢": 5,
    }

    def rank(node: ContentNode) -> int:
        t = node.title.lower()
        for key, val in priority.items():
            if key in t:
                return val
        return 3

    return sorted(sections, key=rank)


def _pop_abstract(sections: list[ContentNode]) -> ContentNode | None:
    for i, s in enumerate(sections):
        t = s.title.lower()
        if "abstract" in t or "摘要" in t:
            return sections.pop(i)
    return None


def _collect_sections(node: ContentNode) -> list[ContentNode]:
    if node.children:
        return list(node.children)
    return [node]


def _is_reference(t: str) -> bool:
    return any(k in t for k in ("reference", "acknowledg", "参考文献", "致谢"))


def _is_result(t: str) -> bool:
    return any(k in t for k in ("result", "discussion", "结果", "讨论"))


def _is_experiment(t: str) -> bool:
    return any(k in t for k in ("experiment", "evaluation", "实验"))


def _is_method(t: str) -> bool:
    return any(k in t for k in ("method", "approach", "方法", "framework", "模型", "architecture", "model"))


def _is_intro(t: str) -> bool:
    return any(k in t for k in ("intro", "introduction", "引言", "背景"))


def _section_accent(title: str) -> tuple[int, int, int]:
    t = title.lower()
    if _is_reference(t):
        return (100, 110, 125)
    return (41, 84, 144)


def _section_icon(title: str) -> str:
    t = title.lower()
    if "abstract" in t or "摘要" in t:
        return "doc"
    if "intro" in t or "引言" in t:
        return "lightbulb"
    if "method" in t or "方法" in t:
        return "gear"
    if "experiment" in t or "result" in t or "实验" in t:
        return "chart"
    if "conclusion" in t or "结论" in t:
        return "flag"
    return "doc"
