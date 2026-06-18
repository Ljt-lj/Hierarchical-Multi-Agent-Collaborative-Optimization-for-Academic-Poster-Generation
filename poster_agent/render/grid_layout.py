"""学术海报网格布局：摘要通栏 + 多栏 masonry，无重叠、等间距."""

from __future__ import annotations

from dataclasses import dataclass

import re

from poster_agent.config import PosterConfig
from poster_agent.models.trees import ContentNode, LayoutRect, PosterNode

# 区块之间统一间距（布局矩形边缘到边缘）
BLOCK_GAP = 16
HEADER_H = 280
FOOTER_H = 50
LANDSCAPE_HEADER_H = 200
LANDSCAPE_FOOTER_H = 20
INNER_PAD = 20


def _is_landscape(config: PosterConfig) -> bool:
    return config.width > config.height


def _layout_chrome(config: PosterConfig) -> tuple[int, int]:
    if _is_landscape(config) and config.academic_style:
        return LANDSCAPE_HEADER_H, LANDSCAPE_FOOTER_H
    return HEADER_H, FOOTER_H


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
        if self.config.three_column_layout and self.config.academic_style:
            return self._layout_three_column(content, height_boost)
        return self._layout_masonry(content, height_boost)

    def _layout_masonry(self, content: ContentNode, height_boost: dict[str, float] | None = None) -> PosterNode:
        sections = _collect_sections(content)
        if not sections:
            sections = [content]

        ordered = _order_sections(sections)
        abstract = _pop_abstract(ordered)
        body_sections = ordered

        content_x = self.config.margin
        content_y = HEADER_H + BLOCK_GAP
        content_w = self.config.width - 2 * self.config.margin
        content_bottom = self.config.height - FOOTER_H - BLOCK_GAP

        children: list[PosterNode] = []

        if abstract:
            plan = _plan_section(abstract, content_w, self.config, full_width=True)
            abs_cap = 0.22 if len(body_sections) >= 4 else 0.28
            h = min(plan.estimated_h, int((content_bottom - content_y) * abs_cap))
            h = max(h, plan.estimated_h // 2)
            rect = LayoutRect(content_x, content_y, content_w, h)
            children.append(_make_node(abstract, rect, plan))
            body_start_y = content_y + h + BLOCK_GAP
        else:
            body_start_y = content_y

        if not body_sections:
            return _wrap_root(content, children, content_x, content_y, content_w, content_bottom - content_y)

        num_cols = 3 if len(body_sections) >= 7 else 2
        col_w = (content_w - BLOCK_GAP * (num_cols - 1)) // num_cols
        available_h = content_bottom - body_start_y

        plans = [_plan_section(s, col_w, self.config) for s in body_sections]
        weights = [_section_layout_weight(s) for s in body_sections]
        total_est = sum(p.estimated_h * w for p, w in zip(plans, weights)) + BLOCK_GAP * (len(plans) // num_cols)
        scale = min(1.0, available_h / max(total_est, 1))
        if len(body_sections) <= 5 and scale > 0.72:
            scale = min(1.0, scale * 1.12)

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
        _fill_sparse_sections(placed, body_sections)
        _stretch_column_tails(placed, content_bottom, body_sections)

        return _wrap_root(content, children, content_x, content_y, content_w, content_bottom - content_y)

    def _layout_three_column(
        self, content: ContentNode, height_boost: dict[str, float] | None = None,
    ) -> PosterNode:
        """横版三栏：左=引言+方法(25%)，中=结果(50%)，右=讨论(25%)."""
        sections = _collect_sections(content)
        if not sections:
            sections = [content]

        ordered = _order_sections(sections)
        abstract = _pop_abstract(ordered) if self.config.merge_abstract_into_intro else None
        if abstract:
            _merge_abstract_into_intro(abstract, ordered)

        body_sections = [s for s in ordered if not _is_reference(s.title.lower())]
        ref_sections = [s for s in ordered if _is_reference(s.title.lower())]
        if not body_sections:
            body_sections = ordered

        header_h, footer_h = _layout_chrome(self.config)
        content_x = self.config.margin
        col_gap = 12 if _is_landscape(self.config) else BLOCK_GAP
        content_y = header_h + col_gap
        content_w = self.config.width - 2 * self.config.margin
        content_bottom = self.config.height - footer_h - col_gap
        fracs = self.config.column_fracs
        gaps = col_gap * 2
        col_ws = [int((content_w - gaps) * f) for f in fracs]
        col_ws[2] = content_w - gaps - col_ws[0] - col_ws[1]
        available_h = content_bottom - content_y

        col_buckets: list[list[ContentNode]] = [[], [], []]
        for sec in body_sections:
            col_buckets[_academic_column(sec.title)].append(sec)
        for sec in ref_sections:
            col_buckets[2].append(sec)

        children: list[PosterNode] = []
        placed: list[tuple[PosterNode, int]] = []
        layout_sections = body_sections + ref_sections

        for col_idx, bucket in enumerate(col_buckets):
            if not bucket:
                continue
            col_w = col_ws[col_idx]
            plans: list[_SectionPlan] = []
            for sec in bucket:
                if sec.children and sec.block_style == "section":
                    plans.append(_plan_section_with_panels(sec, col_w, self.config))
                else:
                    plans.append(_plan_section(sec, col_w, self.config))
            boosts: list[float] = []
            for sec in bucket:
                boost = 1.0
                if height_boost:
                    for key, mult in height_boost.items():
                        if key.lower() in sec.title.lower():
                            boost = max(boost, mult)
                boosts.append(boost)
            heights = _allocate_column_stack(bucket, plans, available_h, col_gap)

            y = content_y
            x = content_x + sum(col_ws[i] + col_gap for i in range(col_idx))
            for plan, sec, h in zip(plans, bucket, heights):
                rect = LayoutRect(x, y, col_w, h)
                node = _make_node(sec, rect, plan, self.config)
                children.append(node)
                placed.append((node, col_idx))
                y += h + col_gap

        _resolve_overlaps(placed, content_bottom)
        return _wrap_root(content, children, content_x, content_y, content_w, content_bottom - content_y)


def _allocate_column_stack(
    bucket: list[ContentNode],
    plans: list[_SectionPlan],
    available_h: int,
    gap: int,
) -> list[int]:
    """按内容估算高度堆叠；仅在溢出时等比压缩，不强行拉伸."""
    n = len(bucket)
    if n == 0:
        return []
    raw = [p.estimated_h for p in plans]
    gap_total = gap * max(n - 1, 0)
    total = sum(raw) + gap_total
    if total > available_h and total > 0:
        scale = (available_h - gap_total) / max(sum(raw), 1)
        raw = [max(int(h * scale), _min_section_height(p)) for h, p in zip(raw, plans)]
    elif total < available_h and raw:
        slack = available_h - total
        raw[-1] += slack
    return raw


def _plan_section_with_panels(
    node: ContentNode,
    col_w: int,
    config: PosterConfig,
) -> _SectionPlan:
    """带 panel 子区块的章节（Results / Methods）高度估算."""
    panel_plans = [_plan_panel_section(p, col_w, config) for p in node.children]
    body_font, title_font, bar_h = _plan_fonts(node, "panel_stack", config)
    chrome = bar_h + INNER_PAD * 2
    panel_gap = 10
    total = chrome + sum(p.estimated_h for p in panel_plans) + panel_gap * max(len(panel_plans) - 1, 0)
    return _SectionPlan(node, "panel_stack", int(total), body_font, title_font, bar_h)


def _plan_panel_section(node: ContentNode, col_w: int, config: PosterConfig) -> _SectionPlan:
    from poster_agent.render.image_fit import is_paper_figure

    layout = _choose_layout(node)
    if node.block_style == "panel":
        layout = "figure_panel" if (node.image_paths or node.visuals) else "text_panel"
    body_font, title_font, _ = _plan_fonts(node, layout, config)
    panel_title_h = 36
    text_h = _text_height(node, col_w, layout, body_font)
    fig_h = _figure_height(node, col_w, layout)
    if layout in ("figure_panel", "figure_top", "figure_rich"):
        paper_boost = 1.35 if any(is_paper_figure(p) for p in node.image_paths if p) else 1.0
        total = int((panel_title_h + fig_h * paper_boost + int(text_h * 0.45) + 12))
    elif layout == "text_panel":
        total = panel_title_h + text_h + 8
    else:
        total = panel_title_h + text_h + fig_h + 8
    return _SectionPlan(node, layout, int(total), body_font, title_font, panel_title_h)


def _allocate_column_heights(
    bucket: list[ContentNode],
    plans: list[_SectionPlan],
    available_h: int,
    config: PosterConfig,
    boosts: list[float],
    *,
    gap: int = BLOCK_GAP,
) -> list[int]:
    """按权重把列内可用高度分完，三栏底对齐、减少列内/列间留白."""
    n = len(bucket)
    if n == 0:
        return []
    gap_total = gap * max(n - 1, 0)
    usable = max(available_h - gap_total, 200)
    weights = [
        _section_layout_weight(sec, config) * (boosts[i] if i < len(boosts) else 1.0)
        for i, sec in enumerate(bucket)
    ]
    total_w = sum(weights) or float(n)
    min_heights = [_min_section_height(p) for p in plans]
    raw = [max(int(usable * w / total_w), min_heights[i]) for i, w in enumerate(weights)]
    total_raw = sum(raw)
    if total_raw > usable:
        scale = usable / total_raw
        raw = [max(int(h * scale), min_heights[i]) for i, h in enumerate(raw)]
        total_raw = sum(raw)
    heights = list(raw)
    slack = usable - total_raw
    if slack > 0 and heights:
        if len(heights) >= 2:
            per = slack // len(heights)
            for i in range(len(heights)):
                heights[i] += per
            heights[-1] += slack - per * len(heights)
        else:
            heights[0] += slack
    elif slack < 0:
        trim = -slack
        for i in sorted(range(len(heights)), key=lambda j: weights[j]):
            cut = min(trim, heights[i] - min_heights[i])
            heights[i] -= cut
            trim -= cut
            if trim <= 0:
                break
    return heights


def _academic_column(title: str) -> int:
    """左=引言+方法，中=结果(宽栏)，右=结论/未来/致谢/参考文献."""
    t = title.lower()
    if _is_result(t) or _is_experiment(t):
        return 1
    if any(k in t for k in (
        "benchmark", "performance", "finding", "dockq", "restraint",
        "quantitative", "structure prediction",
    )):
        return 1
    if _is_discussion(t) or "conclusion" in t or "结论" in t:
        return 2
    if any(k in t for k in ("future", "acknowledg", "reference", "未来", "致谢", "参考")):
        return 2
    if _is_method(t) or "pipeline" in t or "graph construction" in t:
        return 0
    return 0


def _merge_abstract_into_intro(abstract: ContentNode, sections: list[ContentNode]) -> None:
    """将摘要要点并入 Introduction，释放通栏空间."""
    intro = next((s for s in sections if _is_intro(s.title.lower())), None)
    abs_bullets = [b for b in abstract.bullets if b.strip()]
    if not abs_bullets and abstract.summary:
        abs_bullets = [abstract.summary.strip()]
    if intro is None:
        intro = ContentNode(
            title="Introduction",
            summary=abstract.summary or "",
            bullets=abs_bullets[:3],
            weight=0.14,
        )
        sections.insert(0, intro)
        return
    merged: list[str] = []
    seen: set[str] = set()
    for b in abs_bullets[:2] + list(intro.bullets):
        key = b.strip().lower()[:60]
        if key and key not in seen:
            seen.add(key)
            merged.append(b.strip())
    intro.bullets = merged[:4]
    if abstract.summary and not intro.summary:
        intro.summary = abstract.summary[:180]


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


def _make_node(section: ContentNode, rect: LayoutRect, plan: _SectionPlan, config: PosterConfig | None = None) -> PosterNode:
    accent = _section_accent(section.title)
    visual = section.visuals[0] if section.visuals else None
    images = list(section.image_paths)
    if visual and visual.image_path and visual.image_path not in images:
        if not (visual.type == "data_table" and section.image_paths):
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
        figure_captions=list(section.figure_captions),
        table_caption=str(section.paper_tables[0].get("caption", ""))[:120] if section.paper_tables else "",
        layout_mode=plan.layout_mode,
        full_width=plan.full_width,
        block_style=section.block_style,
        children=[_make_panel_node(p, col_w=rect.width, config=config) for p in section.children] if section.children else [],
    )


def _make_panel_node(panel: ContentNode, *, col_w: int, config: PosterConfig | None = None) -> PosterNode:
    cfg = config or PosterConfig()
    plan = _plan_panel_section(panel, col_w, cfg)
    visual = panel.visuals[0] if panel.visuals else None
    images = list(panel.image_paths)
    if visual and visual.image_path and visual.image_path not in images:
        images.insert(0, visual.image_path)
    return PosterNode(
        title=panel.title,
        summary=panel.summary,
        bullets=panel.bullets,
        rect=LayoutRect(0, 0, col_w, plan.estimated_h),
        bg_color=(255, 255, 255),
        accent_color=(41, 84, 144),
        font_size_title=plan.title_font,
        font_size_body=plan.body_font,
        image_paths=images,
        visual_path=visual.image_path if visual else None,
        visual_type=visual.type if visual else None,
        figure_captions=list(panel.figure_captions),
        table_caption=str(panel.paper_tables[0].get("caption", ""))[:120] if panel.paper_tables else "",
        layout_mode=plan.layout_mode,
        block_style="panel",
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
    from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

    layout = _choose_layout(node)
    body_font, title_font, bar_h = _plan_fonts(node, layout, config)
    text_h = _text_height(node, col_w, layout, body_font)
    fig_h = _figure_height(node, col_w, layout)
    chrome = bar_h + INNER_PAD * 3
    if layout == "side_by_side":
        total = chrome + max(text_h, fig_h)
    elif layout == "figure_wrap":
        total = chrome + int(max(text_h * 0.62, fig_h) + text_h * 0.38)
        if node.paper_tables or (node.visuals and node.visuals[0].type == "data_table"):
            total += int(col_w * 0.22)
    elif layout == "figure_rich":
        total = chrome + int(col_w * 1.05 + text_h * 0.20)
    elif layout == "figure_top" and node.image_paths and len(node.image_paths) >= 2:
        total = chrome + int(col_w * 1.35 + text_h * 0.18)
    elif layout == "figure_top" and node.image_paths and (
        is_focus_figure_path(node.image_paths[0]) or is_composite_figure_path(node.image_paths[0])
    ):
        total = chrome + int(col_w * 0.95 + text_h * 0.22)
    elif layout == "figure_top" and node.visuals and node.visuals[0].type in (
        "architecture", "bar_chart", "logic_pipeline",
    ):
        total = chrome + int(text_h * 0.35 + fig_h * 1.15)
    elif layout == "figure_bottom" and node.visuals and node.visuals[0].type == "logic_pipeline":
        total = chrome + int(text_h * 0.32 + fig_h * 1.18)
    else:
        total = chrome + text_h + fig_h
    return _SectionPlan(node, layout, int(total), body_font, title_font, bar_h, full_width)


def _min_section_height(plan: _SectionPlan) -> int:
    return plan.bar_h + INNER_PAD * 3 + 120


def _choose_layout(node: ContentNode) -> str:
    from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

    t = node.title.lower()
    has_paper = bool(node.image_paths)
    has_gen = bool(node.visuals)
    has_fig = has_paper or has_gen
    if not has_fig:
        if _is_reference(t) or "future" in t or "acknowledg" in t:
            return "text_dense"
        if _is_discussion(t) or "conclusion" in t or "结论" in t:
            return "text_dense"
        return "text_only"
    if node.block_style == "panel":
        return "figure_panel"
    if _is_discussion(t):
        return "figure_bottom" if has_gen else "text_dense"
    if _is_result(t) or _is_experiment(t):
        if has_paper and has_gen:
            return "figure_rich"
        return "figure_top"
    if has_gen and _is_intro(t):
        return "figure_top"
    if has_paper and node.image_paths and (
        is_focus_figure_path(node.image_paths[0]) or is_composite_figure_path(node.image_paths[0])
    ):
        return "figure_top"
    if has_paper and not has_gen:
        return "figure_wrap" if not _is_intro(t) else "figure_top"
    if _is_method(t) and has_gen:
        vis = node.visuals[0] if node.visuals else None
        if vis and vis.type == "logic_pipeline":
            return "figure_bottom"
    if "conclusion" in t or "结论" in t:
        return "figure_top"
    if _is_intro(t):
        return "figure_adaptive"
    if _is_method(t):
        return "figure_bottom"
    return "figure_bottom"


def _text_height(node: ContentNode, col_w: int, layout: str, body_font: int) -> int:
    chars_per_line = max(col_w // max(body_font // 2 + 4, 10), 16)
    if layout == "side_by_side":
        chars_per_line = max(col_w // max(body_font + 8, 20), 14)
    elif layout == "figure_adaptive":
        chars_per_line = max(col_w // max(body_font // 2 + 4, 10), 16)
    elif layout == "figure_wrap":
        chars_per_line = max(int(col_w * 0.58) // max(body_font // 2 + 4, 10), 14)
    lines = 0
    if node.summary:
        lines += max(1, len(node.summary) // chars_per_line + 1)
    for b in node.bullets:
        lines += max(1, len(b) // chars_per_line + 1)
    line_h = body_font + 10
    return lines * line_h + 12


def _figure_height(node: ContentNode, col_w: int, layout: str) -> int:
    from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

    if not (node.image_paths or node.visuals):
        return 0
    t = node.title.lower()
    if layout == "figure_rich":
        n_vis = len(node.visuals or [])
        n_paper = len(node.image_paths or [])
        base = col_w * (0.58 + 0.12 * min(n_vis, 2))
        if n_paper:
            base += col_w * 0.28
        return int(base)
    if layout == "figure_wrap" and node.image_paths:
        try:
            from poster_agent.render.image_fit import natural_fit_height
            from PIL import Image
            path = node.image_paths[0]
            src = Image.open(path)
            side_w = int(col_w * 0.34)
            nat = natural_fit_height(src, side_w, is_paper=True)
            return min(int(nat * 1.05), int(col_w * 0.46))
        except Exception:
            return int(col_w * 0.46)
    if node.image_paths:
        n_figs = min(len(node.image_paths), 2)
        ratio = 0.50 if n_figs == 1 else 0.44
        if _is_method(t):
            ratio = 0.52
        if _is_result(t) or _is_experiment(t):
            ratio = 0.55
        return int(col_w * ratio)
    if layout == "figure_top":
        n_figs = min(len(node.image_paths or []) + len(node.visuals or []), 2)
        if node.image_paths and len(node.image_paths) >= 2:
            return int(col_w * 0.68)
        if node.image_paths and (
            is_focus_figure_path(node.image_paths[0]) or is_composite_figure_path(node.image_paths[0])
        ):
            return int(col_w * 0.85)
        if node.visuals and node.visuals[0].type in ("architecture", "bar_chart"):
            return int(col_w * 0.52)
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
        if node.visuals and node.visuals[0].type == "logic_pipeline":
            return int(col_w * 0.58)
        return int(col_w * 0.40)
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


def _content_chars(section: ContentNode | None) -> int:
    if section is None:
        return 200
    return len(section.summary or "") + sum(len(b) for b in section.bullets)


def _fill_sparse_sections(
    placed: list[tuple[PosterNode, int]],
    sections: list[ContentNode],
) -> None:
    """稀疏纯文本区块略缩；含图的区块保留高度以便拉伸后放大插图."""
    title_map = {s.title: s for s in sections}
    for node, _ in placed:
        sec = title_map.get(node.title)
        chars = _content_chars(sec)
        has_fig = bool(sec and (sec.image_paths or sec.visuals))
        if has_fig or chars >= 280:
            continue
        if node.rect.height > 420:
            shrink = min(int(node.rect.height * 0.12), node.rect.height - 320)
            if shrink > 0:
                node.rect = LayoutRect(
                    node.rect.x, node.rect.y, node.rect.width, node.rect.height - shrink
                )


def _section_layout_weight(node: ContentNode, config: PosterConfig | None = None) -> float:
    t = node.title.lower()
    landscape = config is not None and _is_landscape(config)
    heuristic = 1.0
    if _is_intro(t):
        heuristic = 0.95 if (node.image_paths) else 0.72
    elif _is_result(t) or _is_experiment(t):
        heuristic = 2.2 if landscape else 1.35
    elif "conclusion" in t or "结论" in t:
        heuristic = 1.28
    elif _is_method(t):
        heuristic = 1.18 if landscape else 0.95
    elif _is_discussion(t):
        heuristic = 1.0 if landscape else 0.85
    elif "background" in t or "背景" in t:
        heuristic = 0.92
    content_w = float(node.weight) if node.weight and node.weight > 0 else 1.0
    density = min(_content_chars(node) / 900, 1.4)
    fig_boost = 0.15 if (node.image_paths or node.visuals) else 0.0
    return (0.45 * heuristic + 0.35 * content_w + 0.20 * density + fig_boost)


def _section_stretch_weight(node: ContentNode | None) -> float:
    if node is None:
        return 1.0
    t = node.title.lower()
    if _is_intro(t):
        return 1.0 if (node.image_paths or node.visuals) else 0.7
    if _is_discussion(t):
        return 1.2
    if _is_method(t):
        return 1.15 if (node.visuals) else 0.9
    if _is_result(t) or _is_experiment(t):
        return 1.4
    if "conclusion" in t or "结论" in t:
        return 1.1
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
        "future": 5, "acknowledg": 6, "reference": 7, "参考文献": 7, "致谢": 6,
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


def _is_discussion(t: str) -> bool:
    return bool(re.search(r"\bdiscussion\b", t) or "讨论" in t)


def _is_result(t: str) -> bool:
    if _is_discussion(t):
        return False
    return bool(re.search(r"\bresults?\b", t) or "结果" in t)


def _is_experiment(t: str) -> bool:
    return any(k in t for k in ("experiment", "evaluation", "实验"))


def _is_method(t: str) -> bool:
    return any(k in t for k in ("method", "approach", "方法", "framework", "模型", "architecture", "model"))


def _is_intro(t: str) -> bool:
    t = t.lower()
    if re.search(r"\bdiscussion\b", t):
        return False
    return bool(
        re.search(r"\b(introduction|intro)\b", t)
        or "引言" in t
        or ("背景" in t and "discussion" not in t)
    )


def _section_accent(title: str) -> tuple[int, int, int]:
    t = title.lower()
    if _is_reference(t):
        return (100, 110, 125)
    return (41, 84, 144)


def _section_icon(title: str) -> str:
    t = title.lower()
    if "abstract" in t or "摘要" in t:
        return "doc"
    if _is_intro(t):
        return "lightbulb"
    if "method" in t or "方法" in t:
        return "gear"
    if "experiment" in t or "result" in t or "实验" in t:
        return "chart"
    if "conclusion" in t or "结论" in t:
        return "flag"
    return "doc"
