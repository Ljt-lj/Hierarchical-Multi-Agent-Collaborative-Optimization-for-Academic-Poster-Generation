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
from poster_agent.agents.logic_planner_agent import LogicPlannerAgent
from poster_agent.agents.parser_agent import ParserAgent
from poster_agent.agents.painter_agent import PainterAgent
from poster_agent.agents.poster_error_analyzer import PosterErrorAnalyzer
from poster_agent.agents.refiner_agent import RefinerAgent
from poster_agent.agents.semantic_agent import SemanticCheckAgent
from poster_agent.agents.visual_agent import VisualAgent
from poster_agent.config import Config
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, EvaluationScore, PosterNode, RawNode
from poster_agent.models.render_report import PosterRenderReport
from poster_agent.render.language import detect_from_raw_tree
from poster_agent.utils.output_paths import (
    configure_paper_output,
    paper_slug_from_source,
)

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
    output_dir: Path = field(default_factory=Path)
    paper_slug: str = ""
    output_name: str = "poster"
    best_iteration: int = 1
    iteration_history: list[dict] = field(default_factory=list)


class PosterPipeline:
    def __init__(self, config: Config | None = None):
        self.config = config or Config.load()
        self.llm = LLMClient(self.config.llm)
        self.parser = ParserAgent(self.llm, self.config.poster)
        self.logic_planner = LogicPlannerAgent(self.llm)
        self.refiner = RefinerAgent(self.llm, self.config.poster)
        self.visual = VisualAgent(self.llm, self.config.poster)
        self.semantic = SemanticCheckAgent(self.llm)
        self.layout = LayoutAgent(self.config.poster)
        self.balance = BalanceAgent()
        self.painter = PainterAgent(self.config.output_dir, self.config.poster)
        self.commenter = CommenterAgent(self.llm)
        self.controller = ControllerAgent(self.config.poster)
        self.error_analyzer = PosterErrorAnalyzer()

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

        paper_slug = paper_slug_from_source(
            title=raw_tree.title or "",
            pdf_path=pdf_path if not demo else None,
        )
        paper_out, output_name = configure_paper_output(
            self.config,
            paper_slug=paper_slug,
            output_name=output_name,
        )
        self.painter = PainterAgent(paper_out, self.config.poster)
        console.print(f"[cyan]论文目录: {paper_out}")
        console.print(f"[cyan]输出前缀: {output_name}")

        self._save_json(raw_tree.to_dict(), f"{output_name}_raw_tree.json")
        if raw_tree.tables:
            self._save_json(raw_tree.tables, f"{output_name}_paper_tables.json")
        if raw_tree.figure_catalog:
            self._save_json(raw_tree.figure_catalog, f"{output_name}_figure_catalog.json")

        console.print("  → 逻辑规划智能体")
        logic_plan = self.logic_planner.plan(raw_tree, language=paper_lang)
        self._save_json(
            {
                "core_problem": logic_plan.core_problem,
                "pipeline_steps": logic_plan.pipeline_steps,
                "validation_chain": logic_plan.validation_chain,
                "key_terms": logic_plan.key_terms,
                "poster_sections": logic_plan.poster_sections,
            },
            f"{output_name}_logic_plan.json",
        )

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
            content_tree = self.refiner.refine(
                raw_tree, feedback, language=paper_lang, logic_plan=logic_plan
            )
            self._save_json(content_tree.to_dict(), f"{output_name}_content_iter{iteration}.json")

            if self.config.poster.three_column_layout and self.config.poster.academic_style:
                from poster_agent.agents.section_expander import expand_for_academic_poster

                content = expand_for_academic_poster(
                    content_tree, raw_tree, logic_plan, paper_lang,
                    catalog=index_figure_catalog(raw_tree),
                )

            console.print("  → 可视化智能体")
            from poster_agent.agents.figure_curator import index_figure_catalog

            content_tree = self.visual.enrich(
                content_tree,
                language=paper_lang,
                logic_plan=logic_plan,
                catalog=index_figure_catalog(raw_tree),
            )

            if self.config.poster.three_column_layout and self.config.poster.academic_style:
                from poster_agent.agents.column_balancer import balance_columns

                content_tree = balance_columns(content_tree, self.config.poster)

            self.painter.prepare_visuals(
                content_tree,
                f"{output_name}_iter{iteration}",
                llm=self.llm,
                language=paper_lang,
            )

            console.print("  → 语义检查智能体")
            logic_score, logic_issues = self.semantic.check(content_tree, language=paper_lang)

            console.print("  → 布局智能体")
            poster_tree = self.layout.layout(content_tree)
            self.painter.attach_visuals_to_poster(poster_tree, content_tree)
            self._save_json(poster_tree.to_dict(), f"{output_name}_poster_iter{iteration}.json")

            console.print("  → 绘制智能体")
            paper_title = content_tree.title or raw_tree.title
            png_path, pptx_path, poster_tree, render_report = self._paint_with_inner_loop(
                content_tree,
                poster_tree,
                output_name=output_name,
                iteration=iteration,
                paper_title=paper_title,
                authors=getattr(raw_tree, "authors", "") or "",
            )

            console.print("  → 错误分析（GenPilot / Paper2Poster）")
            pre_issues = self.error_analyzer.analyze(render_report, None, content_tree)
            if pre_issues:
                console.print(f"    发现 {len(pre_issues)} 项渲染/内容问题")

            console.print("  → 平衡评估智能体")
            lb_score, balance_metrics = self.balance.evaluate(poster_tree, render_report)

            error_feedback = self.error_analyzer.format_feedback(pre_issues, paper_lang)
            refiner_instructions = self.error_analyzer.format_refiner_instructions(pre_issues, paper_lang)

            console.print("  → 多模态评论智能体")
            sc, itm, comment_feedback = self.commenter.evaluate(
                poster_tree,
                png_path,
                balance_metrics,
                language=paper_lang,
                render_issues=error_feedback,
            )

            console.print("  → 全局控制智能体")
            final_score, should_stop = self.controller.decide(
                sc, lb_score, itm, logic_score, iteration, comment_feedback
            )

            record = {
                "iteration": iteration,
                "logic_score": logic_score,
                "balance_metrics": balance_metrics,
                "render_issues": len(pre_issues),
                "overflow_sections": render_report.overflow_sections,
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

            feedback = self.controller.build_refiner_feedback(
                final_score,
                logic_issues,
                error_feedback=error_feedback,
                refiner_instructions=refiner_instructions,
            )

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
            output_dir=paper_out,
            paper_slug=paper_slug,
            output_name=output_name,
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

    def _paint_with_inner_loop(
        self,
        content_tree: ContentNode,
        poster_tree: PosterNode,
        *,
        output_name: str,
        iteration: int,
        paper_title: str,
        authors: str = "",
    ) -> tuple[Path, Path, PosterNode, PosterRenderReport]:
        png_path = Path()
        pptx_path = Path()
        render_report = PosterRenderReport()

        for inner in range(self.config.poster.inner_paint_passes + 1):
            if inner > 0:
                overflow = render_report.overflow_sections
                if not overflow:
                    break
                console.print(
                    f"    Painter–Commenter 内层重绘 {inner}/{self.config.poster.inner_paint_passes} "
                    f"（增高溢出区块: {', '.join(overflow[:3])}）"
                )
                boost = {s: 1.2 for s in overflow}
                poster_tree = self.layout.layout(content_tree, height_boost=boost)
                self.painter.attach_visuals_to_poster(poster_tree, content_tree)

            artifacts = self.painter.paint(
                poster_tree,
                basename=output_name,
                iteration=iteration if inner == 0 else iteration * 10 + inner,
                paper_title=paper_title,
                authors=authors,
            )
            png_path = artifacts["png"]
            pptx_path = artifacts["pptx"]
            render_report = self.painter.renderer.last_render_report

        return png_path, pptx_path, poster_tree, render_report
