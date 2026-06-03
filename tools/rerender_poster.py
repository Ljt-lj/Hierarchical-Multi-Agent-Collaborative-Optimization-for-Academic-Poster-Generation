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
    if raw_path.exists():
        raw_tree = RawNode.from_dict(json.loads(raw_path.read_text(encoding="utf-8")))
        lang = detect_from_raw_tree(raw_tree)
        visual = VisualAgent(LLMClient(config.llm))
        content = visual.enrich(content, raw_tree, language=lang)

    layout = LayoutAgent(config.poster)
    poster = layout.layout(content)

    painter = PainterAgent(out_dir)
    painter.prepare_visuals(content, args.prefix)
    painter.attach_visuals_to_poster(poster, content)

    title = content.title or "Poster"
    paths = painter.paint(poster, basename=args.output_name, paper_title=title)
    print(f"PNG: {paths['png']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
