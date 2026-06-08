#!/usr/bin/env python3
"""从已有 content JSON 快速重新布局并渲染海报（用于验证渲染优化）."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poster_agent.agents.painter_agent import PainterAgent
from poster_agent.agents.layout_agent import LayoutAgent
from poster_agent.agents.visual_agent import VisualAgent
from poster_agent.agents.figure_curator import build_figure_catalog, index_figure_catalog
from poster_agent.agents.logic_planner_agent import LogicPlan
from poster_agent.config import Config
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, RawNode
from poster_agent.render.language import detect_from_raw_tree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("content_json", type=Path)
    parser.add_argument("--raw-json", type=Path, default=None)
    parser.add_argument("--output-name", default="rerender")
    parser.add_argument("--prefix", default="rerender")
    args = parser.parse_args()

    content = ContentNode.from_dict(json.loads(args.content_json.read_text(encoding="utf-8")))
    out_dir = args.content_json.parent
    config = Config.load()
    config.output_dir = out_dir

    raw_path = args.raw_json or out_dir / args.content_json.name.replace("_content_", "_raw_").replace("_iter4", "_iter4")
    if raw_path.name.endswith("_content_iter4.json"):
        raw_path = out_dir / raw_path.name.replace("_content_iter4", "_raw_tree")
    if not raw_path.exists():
        raw_path = out_dir / args.content_json.name.replace("content", "raw").replace("_iter4.json", "_tree.json")
    raw_tree = None
    lang = "en"
    logic_plan = None
    catalog = None
    if raw_path.exists():
        raw_tree = RawNode.from_dict(json.loads(raw_path.read_text(encoding="utf-8")))
        lang = detect_from_raw_tree(raw_tree)
        catalog = index_figure_catalog(raw_tree)
        prefix = args.content_json.stem.split("_content_")[0]
        logic_path = out_dir / f"{prefix}_logic_plan.json"
        if logic_path.exists():
            logic_plan = LogicPlan(**json.loads(logic_path.read_text(encoding="utf-8")))
        pdf_path = ROOT / "s41592-025-02820-1.pdf"
        if not pdf_path.exists():
            for p in out_dir.parent.rglob("s41592-025-02820-1.pdf"):
                pdf_path = p
                break
        if pdf_path.exists():
            from poster_agent.agents.figure_curator import extract_figure_captions
            from poster_agent.extract.figure_page_renderer import (
                render_figure_composites,
                render_figure_focus_panels,
            )
            import fitz

            doc = fitz.open(pdf_path)
            text = "\n".join(page.get_text("text") for page in doc)
            doc.close()
            caps = extract_figure_captions(text)
            image_dir = pdf_path.parent / f"{pdf_path.stem}_images"
            for item in render_figure_focus_panels(pdf_path, image_dir, caps):
                catalog[Path(item.path).name] = item.to_dict()
                catalog[item.path.replace("\\", "/")] = item.to_dict()
            for item in render_figure_composites(pdf_path, image_dir, caps):
                key = Path(item.path).name
                if key not in catalog:
                    catalog[key] = item.to_dict()
                    catalog[item.path.replace("\\", "/")] = item.to_dict()
        from poster_agent.agents.figure_curator import normalize_poster_figure_paths

        normalize_poster_figure_paths(content, catalog=catalog)
        visual = VisualAgent(LLMClient(config.llm), config.poster)
        if config.poster.three_column_layout and config.poster.academic_style:
            from poster_agent.agents.section_expander import expand_for_academic_poster

            content = expand_for_academic_poster(content, raw_tree, logic_plan, lang, catalog=catalog)
        content = visual.enrich(content, language=lang, logic_plan=logic_plan, catalog=catalog)
        if config.poster.three_column_layout and config.poster.academic_style:
            from poster_agent.agents.column_balancer import balance_columns

            content = balance_columns(content, config.poster)

    layout = LayoutAgent(config.poster)
    poster = layout.layout(content)

    painter = PainterAgent(out_dir, config.poster)
    painter.prepare_visuals(content, args.prefix, llm=LLMClient(config.llm), language=lang)
    painter.attach_visuals_to_poster(poster, content)

    title = content.title or "Poster"
    authors = getattr(raw_tree, "authors", "") if raw_tree else ""
    paths = painter.paint(poster, basename=args.output_name, paper_title=title, authors=authors)
    print(f"PNG: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
