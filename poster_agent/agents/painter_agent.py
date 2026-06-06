"""绘制智能体：生成可视化资产并渲染海报."""

from __future__ import annotations

from pathlib import Path

from poster_agent.agents.visual_evaluator import VisualEvaluator
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, PosterNode
from poster_agent.render.poster_renderer import PosterRenderer
from poster_agent.render.visual_generator import VisualGenerator
from poster_agent.render.visual_registry import VisualRegistry, poster_sections


class PainterAgent:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.renderer = PosterRenderer(output_dir)
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
            spec = node.visuals[0]
            best_spec = spec
            best_score = -1.0

            for attempt in range(VisualEvaluator.MAX_ATTEMPTS):
                name = f"{prefix}_s{i}" if attempt == 0 else f"{prefix}_s{i}_r{attempt}"
                path = self.visual_gen.generate(spec, name)
                if path:
                    spec.image_path = str(path)

                score, issues = evaluator.evaluate_spec(spec, node, registry, language)
                img_issues = evaluator.evaluate_image(path, spec)
                combined = score - 0.08 * len(img_issues)

                if combined > best_score:
                    best_score = combined
                    best_spec = spec

                if combined >= VisualEvaluator.PASS_SCORE and not img_issues:
                    break

                if attempt < VisualEvaluator.MAX_ATTEMPTS - 1:
                    spec = evaluator.refine_spec(
                        spec, node, issues + img_issues, registry, language
                    )

            node.visuals[0] = best_spec
            registry.register(best_spec)

    def attach_visuals_to_poster(self, poster_tree: PosterNode, content_tree: ContentNode) -> None:
        """将可视化与论文插图同步到海报树."""
        content_map = _index_content(content_tree)

        def walk(node: PosterNode) -> None:
            key = node.title.lower()
            for ck, cv in content_map.items():
                if _match(key, ck):
                    if cv.visuals and cv.visuals[0].image_path:
                        node.visual_path = cv.visuals[0].image_path
                        node.visual_type = cv.visuals[0].type
                    imgs = list(cv.image_paths)
                    if cv.visuals:
                        for v in cv.visuals:
                            if v.image_path and v.image_path not in imgs:
                                imgs.insert(0, v.image_path)
                    node.image_paths = imgs
                    node.image_path = imgs[0] if imgs else node.image_path
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
    ) -> dict[str, Path]:
        suffix = f"_iter{iteration}" if iteration else ""
        name = f"{basename}{suffix}"
        png_path = self.renderer.render_png(poster_tree, name, paper_title)
        pptx_path = self.renderer.render_pptx(poster_tree, name, paper_title)
        return {"png": png_path, "pptx": pptx_path}


def _index_content(node: ContentNode) -> dict[str, ContentNode]:
    out: dict[str, ContentNode] = {}

    def walk(n: ContentNode) -> None:
        out[n.title.lower()] = n
        for c in n.children:
            walk(c)

    walk(node)
    return out


def _match(a: str, b: str) -> bool:
    return a in b or b in a
