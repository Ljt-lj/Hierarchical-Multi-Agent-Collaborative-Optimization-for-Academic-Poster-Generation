#!/usr/bin/env python3
"""批量实验：演示模式 + 可选 PDF 目录批量处理."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console

from poster_agent.config import Config
from poster_agent.pipeline import PosterPipeline

console = Console()


def run_batch(pdf_dir: Path | None, num_demo: int, output_dir: Path) -> dict:
    config = Config.load()
    config.output_dir = output_dir
    pipeline = PosterPipeline(config)

    results = []
    if pdf_dir and pdf_dir.exists():
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        console.print(f"发现 {len(pdfs)} 篇 PDF")
        for i, pdf in enumerate(pdfs):
            name = f"paper_{i:03d}_{pdf.stem}"
            try:
                r = pipeline.run(pdf_path=pdf, output_name=name)
                results.append(_summarize(name, r))
            except Exception as e:
                console.print(f"[red]失败 {pdf.name}: {e}")
    else:
        console.print(f"运行 {num_demo} 次演示实验")
        for i in range(num_demo):
            name = f"demo_{i:03d}"
            r = pipeline.run(demo=True, output_name=name)
            results.append(_summarize(name, r))

    report = _aggregate(results)
    report_path = output_dir / "experiment_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_report(report)
    return report


def _summarize(name: str, result) -> dict:
    s = result.final_score
    return {
        "name": name,
        "iterations": len(result.iteration_history),
        "semantic_completeness": s.semantic_completeness,
        "layout_balance": s.layout_balance,
        "image_text_match": s.image_text_match,
        "overall": s.overall,
        "png": str(result.png_path),
    }


def _aggregate(results: list[dict]) -> dict:
    if not results:
        return {"count": 0, "results": []}

    def avg(key: str) -> float:
        return statistics.mean(r[key] for r in results)

    return {
        "timestamp": datetime.now().isoformat(),
        "count": len(results),
        "metrics": {
            "semantic_completeness_avg": round(avg("semantic_completeness"), 3),
            "layout_balance_avg": round(avg("layout_balance"), 3),
            "image_text_match_avg": round(avg("image_text_match"), 3),
            "overall_avg": round(avg("overall"), 3),
            "iteration_count_avg": round(avg("iterations"), 2),
        },
        "results": results,
    }


def _print_report(report: dict) -> None:
    console.rule("[bold]实验报告")
    if report["count"] == 0:
        console.print("无结果")
        return
    m = report["metrics"]
    console.print(f"样本数: {report['count']}")
    console.print(f"语义完整性 (SC): {m['semantic_completeness_avg']:.3f}")
    console.print(f"布局平衡分 (LB): {m['layout_balance_avg']:.3f}")
    console.print(f"图文匹配度:      {m['image_text_match_avg']:.3f}")
    console.print(f"综合评分:        {m['overall_avg']:.3f}")
    console.print(f"平均迭代次数:    {m['iteration_count_avg']:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="批量实验运行器")
    parser.add_argument("--pdf-dir", type=Path, help="PDF 论文目录")
    parser.add_argument("--demo-count", type=int, default=3, help="演示模式运行次数")
    parser.add_argument("--output", type=Path, default=Path("outputs/experiments"))
    args = parser.parse_args()

    config = Config.load()
    if not config.llm.api_key:
        print("错误：未找到 API 密钥", file=sys.stderr)
        return 1

    args.output.mkdir(parents=True, exist_ok=True)
    run_batch(args.pdf_dir, args.demo_count, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
