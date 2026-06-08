"""绘制智能体：生成可视化资产并渲染海报."""

from __future__ import annotations

from pathlib import Path

import re

from poster_agent.config import PosterConfig
from poster_agent.agents.visual_evaluator import VisualEvaluator
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, PosterNode
from poster_agent.render.poster_renderer import PosterRenderer
from poster_agent.render.visual_generator import VisualGenerator
from poster_agent.render.visual_registry import VisualRegistry, poster_sections


class PainterAgent:
    def __init__(self, output_dir: Path, poster_config: PosterConfig | None = None):
        self.output_dir = output_dir
        self.config = poster_config or PosterConfig()
        self.renderer = PosterRenderer(
            output_dir, academic_style=self.config.academic_style, poster_config=self.config,
        )
        self.visual_gen = VisualGenerator(output_dir)

    def prepare_visuals(
        self,
        content_tree: ContentNode,
        prefix: str,
        *,
        llm: LLMClient | None = None,
        language: str = "en",
    ) -> None:
        """为海报顶层区块生成插图，评估质量后必要时优化重生成."""
        evaluator = VisualEvaluator(llm)
        registry = VisualRegistry()

        for i, node in enumerate(poster_sections(content_tree)):
            if not node.visuals:
                continue
            updated: list = []
            for j, spec in enumerate(node.visuals):
                best_spec = spec
                best_score = -1.0
                best_path: Path | None = None
                lock_spec = _lock_visual_spec(node, spec)
                for attempt in range(VisualEvaluator.MAX_ATTEMPTS):
                    name = f"{prefix}_s{i}v{j}" if attempt == 0 else f"{prefix}_s{i}v{j}_r{attempt}"
                    path = self.visual_gen.generate(spec, name)
                    if path:
                        spec.image_path = str(path)
                    score, issues = evaluator.evaluate_spec(spec, node, registry, language)
                    img_issues = evaluator.evaluate_image(path, spec)
                    combined = score - 0.08 * len(img_issues)
                    if combined > best_score:
                        best_score = combined
                        best_spec = spec
                        best_path = path
                    if lock_spec or (combined >= VisualEvaluator.PASS_SCORE and not img_issues):
                        break
                    if attempt < VisualEvaluator.MAX_ATTEMPTS - 1:
                        spec = evaluator.refine_spec(
                            spec, node, issues + img_issues, registry, language
                        )
                if best_path:
                    best_spec.image_path = str(best_path)
                updated.append(best_spec)
                registry.register(best_spec)
            node.visuals = updated

    def attach_visuals_to_poster(self, poster_tree: PosterNode, content_tree: ContentNode) -> None:
        """将可视化与论文插图同步到海报树."""
        content_map = _index_content(content_tree)

        def walk(node: PosterNode) -> None:
            if node.layout_mode == "panel_stack" or (
                node.children and any(c.block_style == "panel" for c in node.children)
            ):
                for c in node.children:
                    walk(c)
                return
            key = node.title.lower()
            for ck, cv in content_map.items():
                if _match(key, ck):
                    imgs: list[str] = []
                    for p in cv.image_paths:
                        if p and p not in imgs:
                            imgs.append(p)
                    for v in cv.visuals or []:
                        if v.image_path and v.image_path not in imgs:
                            imgs.append(v.image_path)
                    node.image_paths = imgs
                    node.image_path = imgs[0] if imgs else node.image_path
                    if cv.visuals and cv.visuals[0].image_path:
                        node.visual_path = cv.visuals[0].image_path
                        node.visual_type = cv.visuals[0].type
                    node.layout_mode = _resolve_layout_mode(cv, imgs)
                    break
            for c in node.children:
                walk(c)

        walk(poster_tree)

    def paint(
        self,
        poster_tree: PosterNode,
        *,
        basename: str = "poster",
        iteration: int = 0,
        paper_title: str | None = None,
        authors: str | None = None,
    ) -> dict[str, Path]:
        suffix = f"_iter{iteration}" if iteration else ""
        name = f"{basename}{suffix}"
        png_path = self.renderer.render_png(poster_tree, name, paper_title, authors)
        pptx_path = self.renderer.render_pptx(poster_tree, name, paper_title, authors)
        return {"png": png_path, "pptx": pptx_path}


def _lock_visual_spec(node: ContentNode, spec) -> bool:
    """数据驱动的基准图不再经 LLM 精炼，避免被删成单柱."""
    t = node.title.lower()
    if spec.type == "bar_chart" and "dockq" in t and "comparison" in t:
        labels = spec.data.get("labels") or []
        values = spec.data.get("values") or []
        return len(labels) >= 2 and len(values) >= 2
    return False


def _index_content(node: ContentNode) -> dict[str, ContentNode]:
    out: dict[str, ContentNode] = {}

    def walk(n: ContentNode) -> None:
        out[n.title.lower()] = n
        for c in n.children:
            walk(c)

    walk(node)
    return out


def _is_discussion_title(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("discussion", "讨论"))


def _is_results_title(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("result", "experiment", "evaluation", "结果", "实验"))


def _is_method_title(title: str) -> bool:
    t = title.lower()
    return any(k in t for k in ("method", "approach", "framework", "模型", "方法", "架构", "pipeline"))


def _is_intro_title(title: str) -> bool:
    t = title.lower()
    if re.search(r"\bdiscussion\b", t):
        return False
    return bool(
        re.search(r"\b(introduction|intro)\b", t)
        or "引言" in t
        or ("背景" in t and "discussion" not in t)
    )


def _resolve_layout_mode(section: ContentNode, image_paths: list[str]) -> str:
    from poster_agent.render.grid_layout import _choose_layout

    probe = ContentNode(
        title=section.title,
        summary=section.summary or "",
        bullets=list(section.bullets),
        visuals=list(section.visuals or []),
        image_paths=[p for p in image_paths if p],
        paper_tables=list(section.paper_tables or []),
    )
    return _choose_layout(probe)


def _match(a: str, b: str) -> bool:
    return a in b or b in a
