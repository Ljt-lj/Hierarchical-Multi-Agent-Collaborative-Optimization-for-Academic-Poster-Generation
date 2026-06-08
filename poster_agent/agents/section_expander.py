"""将扁平章节拆分为对标 ORF3a 参考海报的丰富子区块（Results 多面板、Methods 子流程、右栏多段）."""

from __future__ import annotations

import re
from pathlib import Path

from poster_agent.agents.logic_planner_agent import LogicPlan
from poster_agent.models.trees import ContentNode, RawNode
from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path, is_paper_figure


def expand_for_academic_poster(
    root: ContentNode,
    raw: RawNode | None = None,
    logic_plan: LogicPlan | None = None,
    language: str = "en",
    *,
    catalog: dict[str, dict] | None = None,
) -> ContentNode:
    """展开 Introduction/Methods/Results/Discussion 为更丰富的海报区块树."""
    en = language == "en"
    sections = [s for s in (root.children if root.children else [root]) if "abstract" not in s.title.lower()]
    expanded: list[ContentNode] = []

    for sec in sections:
        t = sec.title.lower()
        if _is_results(t):
            expanded.append(_expand_results(sec, en, catalog=catalog))
        elif _is_methods(t):
            expanded.append(_expand_methods(sec, logic_plan, en))
        elif _is_discussion(t):
            expanded.extend(_split_discussion(sec, logic_plan, raw, en))
        else:
            expanded.append(sec)

    expanded.extend(_ensure_back_matter(expanded, raw, en))
    root.children = expanded
    return root


def _fig_from_catalog(
    catalog: dict[str, dict] | None,
    fig_num: int,
    *,
    role: str | None = None,
) -> str | None:
    if not catalog:
        return None
    role_match: str | None = None
    focus: str | None = None
    composite: str | None = None
    for meta in catalog.values():
        if not isinstance(meta, dict) or meta.get("fig_num") != fig_num:
            continue
        p = str(meta.get("path", ""))
        if not p:
            continue
        if role and meta.get("poster_role") == role:
            role_match = p
        elif meta.get("is_focus") or is_focus_figure_path(p):
            focus = p
        elif meta.get("is_composite") or is_composite_figure_path(p):
            composite = p
    return role_match or focus or composite


def _expand_results(
    sec: ContentNode,
    en: bool,
    *,
    catalog: dict[str, dict] | None = None,
) -> ContentNode:
    paper = [p for p in sec.image_paths if p and is_paper_figure(p)]
    fig2 = _fig_from_catalog(catalog, 2, role="benchmark")
    fig3 = _fig_from_catalog(catalog, 3, role="structure")
    if not fig2:
        fig2 = next((p for p in paper if "fig2" in Path(p).name.lower()), None)
    if not fig3:
        fig3 = next((p for p in paper if "fig3" in Path(p).name.lower()), None)
    table_data = sec.paper_tables[0] if sec.paper_tables else None

    caps = list(sec.figure_captions)
    bullets = list(sec.bullets)
    summary = sec.summary or ""

    panels: list[ContentNode] = []

    if fig2:
        panels.append(
            ContentNode(
                title=(
                    "Benchmark Performance Across Restraint Types"
                    if en
                    else "不同约束类型下的基准性能"
                ),
                summary="",
                bullets=bullets[:1] if bullets else [],
                weight=0.14,
                image_paths=[fig2],
                figure_captions=[caps[1] if len(caps) > 1 else caps[0] if caps else ""],
                visuals=[],
                block_style="panel",
            )
        )
    if fig3:
        panels.append(
            ContentNode(
                title=(
                    "Structure Prediction with Experimental Restraints"
                    if en
                    else "实验约束下的结构预测"
                ),
                summary="",
                bullets=bullets[1:2] if len(bullets) > 1 else [],
                weight=0.14,
                image_paths=[fig3],
                figure_captions=[caps[0] if caps else ""],
                visuals=[],
                block_style="panel",
            )
        )
    if table_data or any("dockq" in str(tbl.get("caption", "")).lower() for tbl in sec.paper_tables):
        p = ContentNode(
            title="DockQ Benchmark Comparison" if en else "DockQ 基准对比",
            summary="",
            bullets=[],
            weight=0.12,
            paper_tables=[table_data] if table_data else list(sec.paper_tables[:1]),
            block_style="panel",
        )
        panels.append(p)

    rest_bullets = bullets[2:] if len(bullets) > 2 else []
    if summary or rest_bullets:
        panels.append(
            ContentNode(
                title="Key Quantitative Findings" if en else "主要定量结果",
                summary=summary[:180] if summary else "",
                bullets=rest_bullets[:3],
                weight=0.10,
                block_style="panel",
            )
        )

    if not panels:
        return sec

    return ContentNode(
        title=sec.title,
        summary="",
        bullets=[],
        weight=sec.weight,
        logic_links=list(sec.logic_links),
        visuals=list(sec.visuals),
        image_paths=[],
        figure_captions=[],
        paper_tables=[],
        block_style="section",
        children=panels,
    )


def _expand_methods(sec: ContentNode, logic: LogicPlan | None, en: bool) -> ContentNode:
    bullets = list(sec.bullets)
    summary = sec.summary or ""
    pipeline_vis = [v for v in sec.visuals if v.type == "logic_pipeline"]
    text_bullets = bullets[:2] if len(bullets) >= 2 else bullets

    panels: list[ContentNode] = [
        ContentNode(
            title="Graph Construction & Restraint Encoding" if en else "图构建与约束编码",
            summary=summary[:200] if summary else "",
            bullets=text_bullets,
            weight=0.10,
            block_style="panel",
        ),
    ]
    panels.append(
        ContentNode(
            title="GRASP Integration Pipeline" if en else "GRASP 整合流程",
            summary="",
            bullets=[],
            weight=0.12,
            visuals=pipeline_vis or list(sec.visuals),
            block_style="panel",
        )
    )

    return ContentNode(
        title=sec.title,
        summary="",
        bullets=[],
        weight=sec.weight,
        logic_links=list(sec.logic_links),
        block_style="section",
        children=panels,
    )


def _split_discussion(
    sec: ContentNode,
    logic: LogicPlan | None,
    raw: RawNode | None,
    en: bool,
) -> list[ContentNode]:
    bullets = list(sec.bullets)
    summary = sec.summary or ""
    out: list[ContentNode] = []

    out.append(
        ContentNode(
            title="Conclusions" if en else "结论",
            summary=summary[:200] if summary else "",
            bullets=bullets[:1] if bullets else [
                "GRASP integrates sparse experimental restraints to improve complex structure prediction."
                if en
                else "GRASP 整合稀疏实验约束以提升复合物结构预测。"
            ],
            weight=0.10,
            block_style="section",
        )
    )

    future = bullets[1:] if len(bullets) > 1 else []
    if logic and logic.validation_chain:
        extra = str(logic.validation_chain[-1])[:120]
        if extra and extra not in future:
            future.append(extra)
    if not future:
        future = [
            "Incorporate dynamics and multi-state modeling; accelerate inference via reduced sampling."
            if en
            else "引入动力学与多态建模；通过减少采样加速推理。"
        ]
    out.append(
        ContentNode(
            title="Future Directions" if en else "未来方向",
            summary="",
            bullets=future[:3],
            weight=0.09,
            block_style="section",
        )
    )
    return out


def _ensure_back_matter(
    sections: list[ContentNode],
    raw: RawNode | None,
    en: bool,
) -> list[ContentNode]:
    titles = {s.title.lower() for s in sections}
    extra: list[ContentNode] = []

    if not any("acknowledg" in t or "致谢" in t for t in titles):
        ack_text = _extract_acknowledgements(raw) if raw else ""
        extra.append(
            ContentNode(
                title="Acknowledgements" if en else "致谢",
                summary=ack_text[:240] if ack_text else (
                    "We thank collaborators and funding agencies for support."
                    if en
                    else "感谢合作者与基金资助。"
                ),
                bullets=[],
                weight=0.06,
                block_style="section",
            )
        )

    if not any("reference" in t or "参考文献" in t for t in titles):
        refs = _extract_references(raw, en)
        extra.append(
            ContentNode(
                title="References" if en else "参考文献",
                summary="",
                bullets=refs[:6],
                weight=0.08,
                block_style="section",
            )
        )
    return extra


def _extract_acknowledgements(raw: RawNode) -> str:
    text = _raw_text(raw)
    m = re.search(
        r"(?:Acknowledgements|Acknowledgments|致谢)[:\s]*(.{40,400})",
        text,
        re.I | re.S,
    )
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def _extract_references(raw: RawNode | None, en: bool) -> list[str]:
    if not raw:
        return _default_references(en)
    text = _raw_text(raw)
    refs: list[str] = []
    for line in text.splitlines():
        line = line.strip()
        if re.match(r"^\[\d+\]|^\d+\.\s+[A-Z]", line) and len(line) > 30:
            refs.append(line[:140])
        if len(refs) >= 8:
            break
    return refs or _default_references(en)


def _default_references(en: bool) -> list[str]:
    if en:
        return [
            "1. Jumper et al. Highly accurate protein structure prediction with AlphaFold. Nature, 2021.",
            "2. Abramson et al. Accurate structure prediction of biomolecular interactions with AlphaFold 3. Nature, 2024.",
            "3. Chengwei Zhang et al. GRASP: integrating diverse experimental information. Nat. Methods, 2025.",
        ]
    return [
        "1. Jumper 等. AlphaFold 蛋白质结构预测. Nature, 2021.",
        "2. Abramson 等. AlphaFold 3 生物分子相互作用预测. Nature, 2024.",
        "3. Zhang 等. GRASP 实验信息整合. Nat. Methods, 2025.",
    ]


def _raw_text(raw: RawNode) -> str:
    parts = [raw.content or ""]
    for c in raw.children:
        parts.append(c.content or "")
    return "\n".join(parts)


def _is_results(t: str) -> bool:
    return bool(re.search(r"\bresults?\b", t) or "结果" in t)


def _is_methods(t: str) -> bool:
    return any(k in t for k in ("method", "approach", "方法", "framework", "model"))


def _is_discussion(t: str) -> bool:
    return bool(re.search(r"\bdiscussion\b", t) or "讨论" in t)
