"""多智能体协作主管道."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console
from rich.table import Table

from poster_agent.agents.balance_agent import BalanceAgent
from poster_agent.agents.commenter_agent import CommenterAgent
from poster_agent.agents.controller_agent import ControllerAgent
from poster_agent.agents.layout_agent import LayoutAgent
from poster_agent.agents.parser_agent import ParserAgent
from poster_agent.agents.painter_agent import PainterAgent
from poster_agent.agents.refiner_agent import RefinerAgent
from poster_agent.agents.semantic_agent import SemanticCheckAgent
from poster_agent.agents.visual_agent import VisualAgent
from poster_agent.config import Config
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, EvaluationScore, PosterNode, RawNode
from poster_agent.render.language import detect_from_raw_tree

console = Console()


@dataclass
class PipelineResult:
    raw_tree: RawNode
    content_tree: ContentNode
    poster_tree: PosterNode
    png_path: Path
    pptx_path: Path
    result_png: Path
    result_pptx: Path
    final_score: EvaluationScore
    best_iteration: int = 1
    iteration_history: list[dict] = field(default_factory=list)


class PosterPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        self.llm = LLMClient(self.config.llm)
        self.parser = ParserAgent(self.llm)
        self.refiner = RefinerAgent(self.llm, self.config.poster)
        self.visual = VisualAgent(self.llm)
        self.semantic = SemanticCheckAgent(self.llm)
        self.layout = LayoutAgent(self.config.poster)
        self.balance = BalanceAgent()
        self.painter = PainterAgent(self.config.output_dir)
        self.commenter = CommenterAgent(self.llm)
        self.controller = ControllerAgent(self.config.poster)

    def run(
        self,
        pdf_path: Path | None = None,
        *,
        demo: bool = False,
        output_name: str = "poster",
    ) -> PipelineResult:
        console.rule("[bold blue]学术海报生成管道启动")

        if demo or pdf_path is None:
            console.print("[yellow]使用内置演示论文数据")
            raw_tree = self._demo_raw_tree()
        else:
            console.print(f"[green]解析 PDF: {pdf_path}")
            raw_tree = self.parser.parse(pdf_path)

        paper_lang = detect_from_raw_tree(raw_tree)
        console.print(f"[cyan]论文语言: {'中文' if paper_lang == 'zh' else 'English'}")

        self._save_json(raw_tree.to_dict(), f"{output_name}_raw_tree.json")

        feedback = ""
        content_tree: ContentNode | None = None
        poster_tree: PosterNode | None = None
        png_path = Path()
        pptx_path = Path()
        final_score: EvaluationScore | None = None
        history: list[dict] = []
        best_score = -1.0
        best_iteration = 1
        best_png = Path()
        best_pptx = Path()

        for iteration in range(1, self.config.poster.max_iterations + 1):
            console.rule(f"[bold]迭代 {iteration}")

            console.print("  → 精炼智能体")
            content_tree = self.refiner.refine(raw_tree, feedback, language=paper_lang)
            self._save_json(content_tree.to_dict(), f"{output_name}_content_iter{iteration}.json")

            console.print("  → 可视化智能体")
            content_tree = self.visual.enrich(content_tree, raw_tree, language=paper_lang)
            self.painter.prepare_visuals(content_tree, f"{output_name}_iter{iteration}")

            console.print("  → 语义检查智能体")
            logic_score, logic_issues = self.semantic.check(content_tree, language=paper_lang)

            console.print("  → 布局智能体")
            poster_tree = self.layout.layout(content_tree)
            self.painter.attach_visuals_to_poster(poster_tree, content_tree)
            self._save_json(poster_tree.to_dict(), f"{output_name}_poster_iter{iteration}.json")

            console.print("  → 绘制智能体")
            paper_title = content_tree.title or raw_tree.title
            artifacts = self.painter.paint(
                poster_tree, basename=output_name, iteration=iteration, paper_title=paper_title
            )
            png_path = artifacts["png"]
            pptx_path = artifacts["pptx"]

            console.print("  → 平衡评估智能体")
            lb_score, balance_metrics = self.balance.evaluate(poster_tree)

            console.print("  → 多模态评论智能体")
            sc, itm, comment_feedback = self.commenter.evaluate(
                poster_tree, png_path, balance_metrics, language=paper_lang
            )

            console.print("  → 全局控制智能体")
            final_score, should_stop = self.controller.decide(
                sc, lb_score, itm, logic_score, iteration, comment_feedback
            )

            record = {
                "iteration": iteration,
                "logic_score": logic_score,
                "balance_metrics": balance_metrics,
                **final_score.to_dict(),
            }
            history.append(record)
            self._print_score_table(record)

            if final_score.overall > best_score:
                best_score = final_score.overall
                best_iteration = iteration
                best_png = png_path
                best_pptx = pptx_path

            if should_stop:
                if final_score.overall >= self.config.poster.score_threshold:
                    console.print(
                        f"[bold green]评分达标 ({final_score.overall:.3f} >= {self.config.poster.score_threshold})"
                    )
                else:
                    console.print(f"[yellow]达到最大迭代次数 {self.config.poster.max_iterations}")
                break

            feedback = self.controller.build_refiner_feedback(final_score, logic_issues)

        assert content_tree and poster_tree and final_score
        self._save_json({"history": history, "final": final_score.to_dict()}, f"{output_name}_scores.json")

        result_png = self.config.output_dir / f"{output_name}_result.png"
        result_pptx = self.config.output_dir / f"{output_name}_result.pptx"
        if best_png.exists():
            shutil.copy2(best_png, result_png)
        if best_pptx.exists():
            shutil.copy2(best_pptx, result_pptx)
        # 最佳迭代副本（便于对照）
        if best_png.exists() and best_iteration > 0:
            tagged = self.config.output_dir / f"{output_name}_iter{best_iteration}_result.png"
            shutil.copy2(best_png, tagged)

        console.rule("[bold green]完成")
        console.print(f"最佳迭代: {best_iteration} (score={best_score:.3f})")
        console.print(f"PNG:  {result_png}")
        console.print(f"PPTX: {result_pptx}")

        return PipelineResult(
            raw_tree=raw_tree,
            content_tree=content_tree,
            poster_tree=poster_tree,
            png_path=png_path,
            pptx_path=pptx_path,
            result_png=result_png,
            result_pptx=result_pptx,
            final_score=final_score,
            best_iteration=best_iteration,
            iteration_history=history,
        )

    def _demo_raw_tree(self) -> RawNode:
        return self.parser.parse_text(
            "Hierarchical Multi-Agent Poster Generation",
            {
                "Abstract": (
                    "We propose a training-free hierarchical multi-agent framework "
                    "for academic poster generation with cross-node semantic consistency, "
                    "dynamic layout weighting, and adaptive iteration control."
                ),
                "Introduction": (
                    "Academic posters require concise content and balanced layout. "
                    "Existing methods suffer from logic fragmentation and rigid iteration."
                ),
                "Method": (
                    "Our system uses eight cooperating agents: Parser, Refiner, Semantic Checker, "
                    "Layout, Balance Evaluator, Painter, Multi-Modal Commenter, and Global Controller. "
                    "Content is organized as a weighted binary tree for spatial partitioning."
                ),
                "Experiment": (
                    "We evaluate on 60 papers from NeurIPS/ICML. "
                    "Our method achieves semantic completeness >4.2/5 and layout balance >4.5/5, "
                    "with average iterations ≤2.1, reducing wasted compute by 30%."
                ),
                "Conclusion": (
                    "The proposed framework generates logically coherent and visually balanced posters "
                    "without model training, outperforming Paper2Poster and PosterForest baselines."
                ),
                "References": "Shi et al. 2025; Kim et al. 2025; Xia et al. 2025.",
            },
        )

    def _save_json(self, data: dict, filename: str) -> None:
        path = self.config.output_dir / filename
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _print_score_table(self, record: dict) -> None:
        table = Table(title="评分")
        table.add_column("指标")
        table.add_column("分数")
        for key in ("semantic_completeness", "layout_balance", "image_text_match", "overall"):
            if key in record:
                table.add_row(key, f"{record[key]:.3f}")
        console.print(table)
