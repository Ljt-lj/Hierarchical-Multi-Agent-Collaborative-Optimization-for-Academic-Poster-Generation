"""Re-score saved pipeline iterations and write *_scores.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poster_agent.agents.balance_agent import BalanceAgent
from poster_agent.agents.commenter_agent import CommenterAgent
from poster_agent.agents.controller_agent import ControllerAgent
from poster_agent.agents.poster_error_analyzer import PosterErrorAnalyzer
from poster_agent.agents.semantic_agent import SemanticCheckAgent
from poster_agent.config import Config
from poster_agent.llm_client import LLMClient
from poster_agent.models.trees import ContentNode, PosterNode, RawNode
from poster_agent.render.language import detect_from_raw_tree
from poster_agent.render.poster_renderer import PosterRenderer

OUT = ROOT / "outputs" / "Integrating_diverse_experimental_information_to_assist_prote"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _find_png(out_dir: Path, prefix: str, iteration: int) -> Path | None:
    for name in (
        f"{prefix}_iter{iteration}.png",
        f"{prefix}_iter{iteration}_result.png",
    ):
        p = out_dir / name
        if p.exists():
            return p
    return None


def _render_report(poster: PosterNode, out_dir: Path, config, stem: str) -> object:
    renderer = PosterRenderer(
        out_dir,
        academic_style=config.poster.academic_style,
        poster_config=config.poster,
    )
    renderer.render_png(poster, f"{stem}_rescore_tmp", paper_title=poster.title or "Poster", authors="")
    return renderer.last_render_report


def score_iteration(
    *,
    out_dir: Path,
    prefix: str,
    iteration: int,
    llm: LLMClient,
    config,
    png_override: Path | None = None,
    content_override: Path | None = None,
    poster_override: Path | None = None,
) -> dict:
    content_path = content_override or out_dir / f"{prefix}_content_iter{iteration}.json"
    poster_path = poster_override or out_dir / f"{prefix}_poster_iter{iteration}.json"
    png_path = png_override or _find_png(out_dir, prefix, iteration)
    if not content_path.exists():
        raise FileNotFoundError(content_path)
    if not poster_path.exists():
        raise FileNotFoundError(poster_path)
    if not png_path or not png_path.exists():
        raise FileNotFoundError(f"PNG for iter {iteration} under {prefix}")

    content = ContentNode.from_dict(_load_json(content_path))
    poster = PosterNode.from_dict(_load_json(poster_path))

    raw_path = out_dir / f"{prefix}_raw_tree.json"
    lang = detect_from_raw_tree(RawNode.from_dict(_load_json(raw_path))) if raw_path.exists() else "en"

    semantic = SemanticCheckAgent(llm)
    balance = BalanceAgent()
    commenter = CommenterAgent(llm)
    controller = ControllerAgent(config.poster)
    error_analyzer = PosterErrorAnalyzer()

    logic_score, _ = semantic.check(content, language=lang)
    render_report = _render_report(poster, out_dir, config, png_path.stem)
    pre_issues = error_analyzer.analyze(render_report, None, content)
    lb_score, balance_metrics = balance.evaluate(poster, render_report)
    error_feedback = error_analyzer.format_feedback(pre_issues, lang)
    sc, itm, comment_feedback = commenter.evaluate(
        poster, png_path, balance_metrics, language=lang, render_issues=error_feedback,
    )
    final_score, _ = controller.decide(sc, lb_score, itm, logic_score, iteration, comment_feedback)

    return {
        "iteration": iteration,
        "logic_score": logic_score,
        "balance_metrics": balance_metrics,
        "render_issues": len(pre_issues),
        "overflow_sections": render_report.overflow_sections,
        **final_score.to_dict(),
    }


def rescore_run(prefix: str, out_dir: Path = OUT, extra_pngs: list[tuple[int, Path, Path | None]] | None = None) -> Path:
    config = Config.load()
    config.output_dir = out_dir
    llm = LLMClient(config.llm)

    content_iters = sorted(
        int(p.name.split("_iter")[-1].replace(".json", ""))
        for p in out_dir.glob(f"{prefix}_content_iter*.json")
    )
    if not content_iters and not extra_pngs:
        raise FileNotFoundError(f"No content iterations for {prefix}")

    history: list[dict] = []
    for it in content_iters:
        print(f"Scoring {prefix} iter {it} ...")
        history.append(score_iteration(out_dir=out_dir, prefix=prefix, iteration=it, llm=llm, config=config))

    if extra_pngs:
        content2 = out_dir / f"{prefix}_content_iter{max(content_iters)}.json"
        poster2 = out_dir / f"{prefix}_poster_iter{max(content_iters)}.json"
        for it, png, label in extra_pngs:
            print(f"Scoring {prefix} extra iter {it} ({png.name}) ...")
            history.append(
                score_iteration(
                    out_dir=out_dir,
                    prefix=prefix,
                    iteration=it,
                    llm=llm,
                    config=config,
                    png_override=png,
                    content_override=content2 if content2.exists() else None,
                    poster_override=poster2 if poster2.exists() else None,
                )
            )

    history.sort(key=lambda h: h["iteration"])
    best = max(history, key=lambda h: h["overall"])
    payload = {"history": history, "final": {k: v for k, v in best.items() if k != "iteration"}}
    out_path = out_dir / f"{prefix}_scores.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    for h in history:
        print(
            f"  iter={h['iteration']}  overall={h['overall']:.3f}  "
            f"ITM={h['image_text_match']:.2f}  logic={h['logic_score']:.2f}"
        )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="s41592_02820_v19")
    parser.add_argument("--out-dir", type=Path, default=OUT)
    parser.add_argument(
        "--include-crop",
        action="store_true",
        help="Add v19_crop.png as final iteration score point",
    )
    args = parser.parse_args()

    extra = None
    if args.include_crop:
        crop = args.out_dir / f"{args.prefix}_crop.png"
        if crop.exists():
            max_it = max(
                int(p.name.split("_iter")[-1].replace(".json", ""))
                for p in args.out_dir.glob(f"{args.prefix}_content_iter*.json")
            )
            extra = [(max_it + 1, crop, None)]

    rescore_run(args.prefix, args.out_dir, extra)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
