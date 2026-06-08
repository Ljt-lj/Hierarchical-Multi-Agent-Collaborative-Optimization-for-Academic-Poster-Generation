"""论文原图筛选与章节匹配（复杂论文优先展示 PDF 插图）."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, RawNode
from poster_agent.render.language import language_instruction


@dataclass
class FigureAsset:
    path: str
    page: int
    width: int
    height: int
    score: float
    caption_hint: str = ""
    kind: str = "general"
    fig_num: int = 0
    poster_role: str = ""

    @property
    def area(self) -> int:
        return self.width * self.height

    def to_dict(self) -> dict:
        from poster_agent.extract.figure_cropper import is_poster_crop_path
        from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

        return {
            "path": self.path,
            "page": self.page,
            "width": self.width,
            "height": self.height,
            "score": self.score,
            "caption_hint": self.caption_hint,
            "kind": self.kind,
            "fig_num": self.fig_num,
            "poster_role": self.poster_role,
            "is_composite": is_composite_figure_path(self.path),
            "is_focus": is_focus_figure_path(self.path),
            "is_crop": is_poster_crop_path(self.path),
        }


def catalog_and_filter_images(image_paths: list[str], *, min_area: int = 28_000) -> list[FigureAsset]:
    """按面积筛选有效论文插图，去除图标/小碎片."""
    assets: list[FigureAsset] = []
    for path in image_paths:
        p = Path(path)
        if not p.exists():
            continue
        page = _page_from_name(p.name)
        try:
            with Image.open(p) as img:
                w, h = img.size
        except OSError:
            continue
        area = w * h
        if area < min_area:
            continue
        aspect = w / max(h, 1)
        if aspect > 12 or aspect < 0.08:
            continue
        score = float(area) * (1.2 if area > 120_000 else 1.0)
        assets.append(FigureAsset(str(p), page, w, h, score))
    assets.sort(key=lambda a: (-a.score, a.page))
    assets = _dedupe_by_page(assets)
    _apply_kind_scoring(assets)
    return assets


def _apply_kind_scoring(assets: list[FigureAsset]) -> None:
    """按插图类型与 caption 提升关键数据图/架构图优先级."""
    kind_boost = {
        "data_chart": 2.2,
        "table": 1.9,
        "architecture": 1.7,
        "structure": 1.15,
        "general": 1.0,
    }
    for asset in assets:
        asset.kind = classify_figure(asset)
        mult = kind_boost.get(asset.kind, 1.0)
        cap = (asset.caption_hint or "").lower()
        if cap:
            mult *= 1.12
        if any(w in cap for w in ("dockq", "benchmark", "performance", "comparison", "boxplot", "violin")):
            mult *= 1.25
        if any(w in cap for w in ("scheme", "architecture", "pipeline", "integration")):
            mult *= 1.15
        asset.score *= mult
    assets.sort(key=lambda a: (-a.score, a.page))


def classify_figure(asset: FigureAsset) -> str:
    """分类论文插图：data_chart / architecture / structure / table / general."""
    from poster_agent.render.image_fit import is_composite_figure_path

    cap = (asset.caption_hint or "").lower()
    aspect = asset.width / max(asset.height, 1)

    if any(w in cap for w in ("table", "supplementary table", "汇总", "表格")):
        return "table"
    if any(
        w in cap
        for w in (
            "boxplot", "violin", "dockq", "performance", "benchmark", "comparison",
            "scatter", "heatmap", "plot", "curve", "success rate", "recall",
            "对比", "性能", "基准",
        )
    ):
        return "data_chart"
    if any(
        w in cap
        for w in (
            "scheme", "architecture", "workflow", "pipeline", "model", "framework",
            "overview", "integration", "block", "module", "grasp",
            "架构", "流程", "模型", "示意",
        )
    ):
        return "architecture"
    if any(
        w in cap
        for w in (
            "structure", "3d", "cartoon", "representation", "complex", "binding",
            "结构", "复合物",
        )
    ):
        return "structure"
    if is_composite_figure_path(asset.path) and cap.strip():
        return "general"
    if aspect >= 1.45 and not any(w in cap for w in ("scheme", "structure", "cartoon", "3d")):
        return "data_chart"
    if aspect <= 0.82:
        return "structure"
    return "general"


def extract_figure_captions(text: str) -> dict[int, str]:
    """从正文中提取 Fig/Figure 编号与附近 caption 片段."""
    captions: dict[int, str] = {}
    patterns = [
        r"(?:Fig\.?|Figure)\s*(\d+)[a-z]?\s*[|:.\-–—]?\s*([^\n]{20,220})",
        r"(?:Extended Data Fig\.?|Supplementary Fig\.?)\s*(\d+)[a-z]?\s*[|:.\-–—]?\s*([^\n]{20,180})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            num = int(m.group(1))
            cap = re.sub(r"\s+", " ", m.group(2)).strip()
            if num not in captions or len(cap) > len(captions[num]):
                captions[num] = cap
    return captions


def attach_caption_hints(assets: list[FigureAsset], captions: dict[int, str]) -> None:
    for asset in assets:
        asset.fig_num = _guess_figure_number(asset.page)
        if asset.fig_num in captions:
            asset.caption_hint = captions[asset.fig_num][:200]
        asset.kind = classify_figure(asset)
    if not captions:
        return
    # 同页多图：宽图优先匹配数据类 caption
    by_page: dict[int, list[FigureAsset]] = {}
    for a in assets:
        by_page.setdefault(a.page, []).append(a)
    for page_assets in by_page.values():
        if len(page_assets) <= 1:
            continue
        page_assets.sort(key=lambda a: -(a.width / max(a.height, 1)))
        for i, a in enumerate(page_assets):
            nums = sorted(captions.keys())
            if i < len(nums):
                num = nums[min(i, len(nums) - 1)]
                if not a.caption_hint:
                    a.caption_hint = captions[num][:200]
                    a.fig_num = num
                a.kind = classify_figure(a)


def assign_figures_to_raw_tree(
    raw_tree: RawNode,
    assets: list[FigureAsset],
    llm: LLMClient | None,
    language: str = "en",
    *,
    max_per_section: int = 2,
) -> None:
    """将筛选后的论文原图分配到各章节."""
    if not assets:
        return
    sections = _collect_raw_sections(raw_tree)
    if not sections:
        return

    assignment: dict[str, list[str]] = {}
    if llm and len(assets) <= 40:
        assignment = _llm_assign(sections, assets, llm, language, max_per_section)
    if not assignment:
        assignment = _heuristic_assign(sections, assets, max_per_section)

    assignment = _dedupe_global_assignment(assignment)
    assignment = _enforce_section_figure_policy(assignment)

    for sec in sections:
        paths = assignment.get(sec.title, [])
        if paths:
            sec.images = list(dict.fromkeys(paths + sec.images))[:max_per_section]


def build_figure_catalog(assets: list[FigureAsset]) -> list[dict]:
    return [a.to_dict() for a in assets[:40]]


def index_figure_catalog(raw: RawNode) -> dict[str, dict]:
    """从 raw_tree.figure_catalog 构建 path/name 索引."""
    catalog: dict[str, dict] = {}
    for item in raw.figure_catalog or []:
        path = str(item.get("path", ""))
        if not path:
            continue
        catalog[Path(path).name] = item
        catalog[path.replace("\\", "/")] = item
    return catalog


def _llm_assign(
    sections: list[RawNode],
    assets: list[FigureAsset],
    llm: LLMClient,
    language: str,
    max_per_section: int,
) -> dict[str, list[str]]:
    catalog_lines = []
    for i, a in enumerate(assets[:30]):
        hint = f" | caption: {a.caption_hint[:80]}" if a.caption_hint else ""
        catalog_lines.append(
            f"  fig_{i}: page={a.page}, kind={a.kind}, {a.width}x{a.height}, "
            f"file={Path(a.path).name}{hint}"
        )
    sec_lines = []
    for s in sections:
        preview = (s.content or "")[:400].replace("\n", " ")
        sec_lines.append(f"- {s.title}: {preview}")

    system = (
        "You assign ORIGINAL paper figures to poster sections for a scientific poster. "
        "Prefer: data_chart/benchmark plots → Results; architecture/pipeline schematics → Introduction ONLY; "
        "Methods and Discussion get NO paper figures (Methods uses synthesized pipeline diagram). "
        "Each figure_id may appear ONCE across the entire poster — never reuse. "
        "Abstract: 0 figures. Each section gets 0-2 figures of DIFFERENT kinds when possible. "
        "Output JSON: {assignments: [{section_title, figure_ids: [\"fig_0\", ...], reason}]}. "
        + language_instruction(language)
    )
    if language == "zh":
        system = (
            "为学术海报各章节分配论文原图。方法/架构图给 Method，实验曲线/对比给 Results，"
            "动机/问题示意给 Introduction。每节 0-2 张，仅使用 catalog 中的 fig 编号。"
            "输出 JSON：{assignments: [{section_title, figure_ids, reason}]}。"
            + language_instruction(language)
        )
    user = (
        "Sections:\n" + "\n".join(sec_lines) + "\n\nFigure catalog:\n" + "\n".join(catalog_lines)
    )
    try:
        data = llm.chat_json(system, user, temperature=0.2)
        id_map = {f"fig_{i}": assets[i].path for i in range(min(len(assets), 30))}
        out: dict[str, list[str]] = {}
        for item in data.get("assignments") or []:
            title = item.get("section_title", "")
            paths: list[str] = []
            for fid in item.get("figure_ids") or []:
                p = id_map.get(str(fid))
                if p:
                    paths.append(p)
            if title and paths:
                out[title] = paths[:max_per_section]
        return _dedupe_global_assignment(out)
    except Exception:
        return {}


def _heuristic_assign(
    sections: list[RawNode],
    assets: list[FigureAsset],
    max_per_section: int,
) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    used: set[str] = set()

    def pick(
        n: int,
        *,
        kinds: tuple[str, ...] = (),
        prefer_page: int | None = None,
    ) -> list[str]:
        chosen: list[str] = []
        pool = sorted(assets, key=lambda a: (-a.score, a.page))
        if prefer_page is not None:
            pool = sorted(pool, key=lambda a: (abs(a.page - prefer_page), -a.score))
        if kinds:
            typed = [a for a in pool if a.kind in kinds]
            if typed:
                pool = typed + [a for a in pool if a not in typed]
        for a in pool:
            if a.path in used:
                continue
            chosen.append(a.path)
            used.add(a.path)
            if len(chosen) >= n:
                break
        return chosen

    for sec in sections:
        t = sec.title.lower()
        if any(k in t for k in ("method", "approach", "model", "framework", "方法", "模型", "架构")):
            continue
        elif any(k in t for k in ("result", "experiment", "evaluation", "结果", "实验")):
            paths: list[str] = []
            for fn in (2, 3):
                p = _pick_figure_by_num(assets, fn, used, prefer_composite=True)
                if p and p not in paths:
                    paths.append(p)
            if not paths:
                p = _pick_figure_by_num(assets, 2, used, prefer_composite=True)
                if p:
                    paths.append(p)
            out[sec.title] = paths[:max_per_section]
        elif any(k in t for k in ("intro", "introduction", "背景", "引言")):
            p = _pick_figure_by_num(assets, 1, used, prefer_composite=True)
            out[sec.title] = [p] if p else pick(1, kinds=("architecture", "general"), prefer_page=3)
        elif "abstract" in t or "摘要" in t:
            continue

    return out


def _pick_figure_by_num(
    assets: list[FigureAsset],
    fig_num: int,
    used: set[str],
    *,
    prefer_composite: bool = True,
) -> str | None:
    from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

    candidates = [a for a in assets if a.fig_num == fig_num and a.path not in used]
    for a in candidates:
        if is_focus_figure_path(a.path):
            used.add(a.path)
            return a.path
    for a in candidates:
        if is_composite_figure_path(a.path) and not is_focus_figure_path(a.path):
            used.add(a.path)
            return a.path
    if prefer_composite:
        pass  # already tried composite above
    if candidates:
        used.add(candidates[0].path)
        return candidates[0].path
    return None


def _dedupe_global_assignment(assignment: dict[str, list[str]]) -> dict[str, list[str]]:
    """全局去重：每张论文原图只分配给一个章节."""
    used: set[str] = set()
    out: dict[str, list[str]] = {}
    for title, paths in assignment.items():
        unique: list[str] = []
        for p in paths:
            if p in used:
                continue
            unique.append(p)
            used.add(p)
        if unique:
            out[title] = unique
    return out


def _enforce_section_figure_policy(assignment: dict[str, list[str]]) -> dict[str, list[str]]:
    """按章节角色过滤：Abstract/Methods/Discussion 不使用论文原图."""
    out: dict[str, list[str]] = {}
    for title, paths in assignment.items():
        t = title.lower()
        if any(k in t for k in ("abstract", "摘要", "discussion", "讨论", "conclusion", "结论")):
            continue
        if any(k in t for k in ("method", "approach", "framework", "方法", "模型", "架构")):
            continue
        out[title] = paths
    return out


def _collect_raw_sections(node: RawNode) -> list[RawNode]:
    if node.children:
        out: list[RawNode] = []
        for c in node.children:
            if c.children:
                out.extend(_collect_raw_sections(c))
            else:
                out.append(c)
        return out
    return [node] if node.title else []


def _dedupe_by_page(assets: list[FigureAsset]) -> list[FigureAsset]:
    """同页保留面积最大的若干张."""
    by_page: dict[int, list[FigureAsset]] = {}
    for a in assets:
        by_page.setdefault(a.page, []).append(a)
    kept: list[FigureAsset] = []
    for page_assets in by_page.values():
        page_assets.sort(key=lambda x: -x.score)
        kept.extend(page_assets[:4])
    kept.sort(key=lambda a: (-a.score, a.page))
    return kept


def _page_from_name(name: str) -> int:
    m = re.search(r"page(\d+)", name, re.I)
    return int(m.group(1)) if m else 0


def _guess_figure_number(page: int) -> int:
    """Nature 类论文：Fig 1≈p3, Fig 2≈p4-5, Fig 3≈p6-7."""
    if page <= 2:
        return 0
    if page == 3:
        return 1
    if page <= 5:
        return 2
    if page <= 7:
        return 3
    if page <= 9:
        return 4
    return min(page - 5, 8)


def normalize_poster_figure_paths(
    root: ContentNode,
    *,
    catalog: dict[str, dict] | None = None,
) -> None:
    """全局插图策略：去重、按章节角色分配、Results 优先数据图."""
    from poster_agent.render.visual_registry import poster_sections

    sections = poster_sections(root)
    if not sections and root.children:
        sections = list(root.children)

    used: set[str] = set()

    def no_figure(title: str) -> bool:
        t = title.lower()
        return any(
            k in t
            for k in (
                "abstract", "摘要", "discussion", "讨论", "conclusion", "结论",
                "method", "approach", "framework", "方法", "模型", "架构",
            )
        )

    def is_results(title: str) -> bool:
        t = title.lower()
        return any(k in t for k in ("result", "experiment", "evaluation", "结果", "实验"))

    def is_intro(title: str) -> bool:
        t = title.lower()
        return any(k in t for k in ("intro", "introduction", "背景", "引言"))

    def priority(title: str) -> int:
        if is_results(title):
            return 0
        if is_intro(title):
            return 1
        return 2

    if catalog:
        from poster_agent.render.image_fit import is_composite_figure_path, is_focus_figure_path

        def poster_fig_for(fig_num: int, role: str | None = None) -> str | None:
            """优先 poster_role 裁剪子图，其次 focus，最后 composite."""
            if role:
                for meta in catalog.values():
                    if not isinstance(meta, dict):
                        continue
                    if meta.get("fig_num") != fig_num or meta.get("poster_role") != role:
                        continue
                    p = str(meta.get("path", ""))
                    if p:
                        return p
            focus_match: str | None = None
            composite_match: str | None = None
            for meta in catalog.values():
                if not isinstance(meta, dict) or meta.get("fig_num") != fig_num:
                    continue
                p = str(meta.get("path", ""))
                if not p or p in used:
                    continue
                if meta.get("is_focus") or is_focus_figure_path(p):
                    focus_match = p
                elif meta.get("is_composite") or (
                    is_composite_figure_path(p) and not is_focus_figure_path(p)
                ):
                    composite_match = p
            return focus_match or composite_match

        for sec in sections:
            if _is_paper_benchmark_panel(sec.title):
                p = poster_fig_for(2, "benchmark")
            elif _is_structure_panel(sec.title):
                p = poster_fig_for(3, "structure")
            elif is_intro(sec.title):
                p = poster_fig_for(1, "intro")
            else:
                p = None
            if p:
                sec.image_paths = [p]
                used.add(p)

    for sec in sorted(sections, key=lambda s: priority(s.title)):
        if no_figure(sec.title):
            sec.image_paths = []
            sec.figure_captions = []
            continue
        max_n = 2 if is_results(sec.title) else 1
        unique: list[str] = []
        for p in sec.image_paths:
            if p:
                if p not in used:
                    used.add(p)
                unique.append(p)
        sec.image_paths = unique[:max_n]

    if catalog:
        _refresh_figure_captions(sections, catalog)


def _is_paper_benchmark_panel(title: str) -> bool:
    t = title.lower()
    return "restraint types" in t or ("benchmark performance" in t and "restraint" in t)


def _is_structure_panel(title: str) -> bool:
    t = title.lower()
    return "structure prediction" in t and "restraint" in t


def _catalog_by_kind(catalog: dict[str, dict], kinds: tuple[str, ...]) -> list[dict]:
    seen: set[str] = set()
    items: list[dict] = []
    for meta in catalog.values():
        if not isinstance(meta, dict):
            continue
        p = str(meta.get("path", ""))
        if not p or p in seen or meta.get("kind") not in kinds:
            continue
        seen.add(p)
        items.append(meta)
    items.sort(key=lambda m: (0 if m.get("kind") == "data_chart" else 1, -int(m.get("page", 0))))
    return items


def _refresh_figure_captions(sections: list[ContentNode], catalog: dict[str, dict]) -> None:
    for sec in sections:
        caps: list[str] = []
        for p in sec.image_paths:
            key = Path(p).name
            meta = catalog.get(key) or catalog.get(p.replace("\\", "/"), {})
            hint = str(meta.get("caption_hint", "")).strip()
            fig_num = meta.get("fig_num") or 0
            kind = meta.get("kind", "")
            if hint:
                prefix = f"Fig. {fig_num}" if fig_num else "Fig."
                caps.append(f"{prefix}: {hint[:95]}")
        sec.figure_captions = caps
