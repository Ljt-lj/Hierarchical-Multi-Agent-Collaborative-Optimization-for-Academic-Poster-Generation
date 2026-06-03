#!/usr/bin/env python3
"""批量测试框架：支持 manifest、断点续跑、CSV/HTML 报告."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from poster_agent.config import Config
from poster_agent.pipeline import PosterPipeline

console = Console()


def load_manifest(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    return data.get("papers", [])


def discover_inputs(pdf_dir: Path | None, manifest: Path | None, demo_count: int) -> list[dict]:
    items: list[dict] = []
    if manifest and manifest.exists():
        for entry in load_manifest(manifest):
            items.append({
                "id": entry.get("id", Path(entry["path"]).stem),
                "path": entry.get("path"),
                "demo": entry.get("demo", False),
            })
    elif pdf_dir and pdf_dir.exists():
        for i, pdf in enumerate(sorted(pdf_dir.glob("*.pdf"))):
            items.append({"id": f"paper_{i:03d}_{pdf.stem}", "path": str(pdf), "demo": False})
    else:
        for i in range(demo_count):
            items.append({"id": f"demo_{i:03d}", "path": None, "demo": True})
    return items


def load_completed(output_dir: Path) -> set[str]:
    state = output_dir / "batch_state.json"
    if not state.exists():
        return set()
    data = json.loads(state.read_text(encoding="utf-8"))
    return set(data.get("completed", []))


def save_state(output_dir: Path, completed: set[str], results: list[dict]) -> None:
    state = output_dir / "batch_state.json"
    state.write_text(
        json.dumps({"completed": sorted(completed), "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_batch_test(
    items: list[dict],
    output_dir: Path,
    *,
    resume: bool = False,
    threshold: float | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config.load()
    config.output_dir = output_dir
    if threshold is not None:
        config.poster.score_threshold = threshold

    completed = load_completed(output_dir) if resume else set()
    results: list[dict] = []
    if resume and (output_dir / "batch_state.json").exists():
        prev = json.loads((output_dir / "batch_state.json").read_text(encoding="utf-8"))
        results = prev.get("results", [])

    pipeline = PosterPipeline(config)
    pending = [it for it in items if it["id"] not in completed]

    console.print(f"共 {len(items)} 项，待运行 {len(pending)} 项")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("批量测试", total=len(pending))

        for item in pending:
            item_id = item["id"]
            progress.update(task, description=f"处理 {item_id}")
            t0 = time.time()
            record = {"id": item_id, "status": "ok", "elapsed_sec": 0}

            try:
                if item.get("demo"):
                    r = pipeline.run(demo=True, output_name=item_id)
                else:
                    pdf = Path(item["path"])
                    if not pdf.is_absolute():
                        pdf = ROOT / pdf
                    r = pipeline.run(pdf_path=pdf, output_name=item_id)

                s = r.final_score
                record.update({
                    "elapsed_sec": round(time.time() - t0, 1),
                    "iterations": len(r.iteration_history),
                    "semantic_completeness": round(s.semantic_completeness, 3),
                    "layout_balance": round(s.layout_balance, 3),
                    "image_text_match": round(s.image_text_match, 3),
                    "overall": round(s.overall, 3),
                    "png": str(r.png_path),
                    "pptx": str(r.pptx_path),
                })
                results.append(record)
                completed.add(item_id)
                save_state(output_dir, completed, results)
            except Exception as e:
                record.update({
                    "status": "failed",
                    "elapsed_sec": round(time.time() - t0, 1),
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                })
                results.append(record)
                console.print(f"[red]失败 {item_id}: {e}")

            progress.advance(task)

    report = build_report(results)
    write_reports(output_dir, report, results)
    print_summary(report)
    return report


def build_report(results: list[dict]) -> dict:
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]

    def avg(key: str) -> float:
        vals = [r[key] for r in ok if key in r]
        return sum(vals) / len(vals) if vals else 0.0

    return {
        "timestamp": datetime.now().isoformat(),
        "total": len(results),
        "success": len(ok),
        "failed": len(failed),
        "metrics": {
            "semantic_completeness_avg": round(avg("semantic_completeness"), 3),
            "layout_balance_avg": round(avg("layout_balance"), 3),
            "image_text_match_avg": round(avg("image_text_match"), 3),
            "overall_avg": round(avg("overall"), 3),
            "iteration_count_avg": round(avg("iterations"), 2),
            "elapsed_sec_avg": round(avg("elapsed_sec"), 1),
        },
        "results": results,
    }


def write_reports(output_dir: Path, report: dict, results: list[dict]) -> None:
    (output_dir / "batch_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    csv_path = output_dir / "batch_results.csv"
    if results:
        fields = [
            "id", "status", "overall", "semantic_completeness", "layout_balance",
            "image_text_match", "iterations", "elapsed_sec", "png", "error",
        ]
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(results)

    html = _build_html_report(report, results)
    (output_dir / "batch_report.html").write_text(html, encoding="utf-8")


def _build_html_report(report: dict, results: list[dict]) -> str:
    m = report.get("metrics", {})
    rows = ""
    for r in results:
        img = ""
        if r.get("png") and Path(r["png"]).exists():
            rel = Path(r["png"]).name
            img = f'<img src="{rel}" width="120" />'
        rows += f"""<tr>
          <td>{r.get('id','')}</td>
          <td>{r.get('status','')}</td>
          <td>{r.get('overall','-')}</td>
          <td>{r.get('semantic_completeness','-')}</td>
          <td>{r.get('layout_balance','-')}</td>
          <td>{r.get('iterations','-')}</td>
          <td>{r.get('elapsed_sec','-')}s</td>
          <td>{img}</td>
        </tr>"""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>批量测试报告</title>
<style>
body {{ font-family: sans-serif; margin: 24px; background: #f5f7fa; }}
.card {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,.08); }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
th {{ background: #1c3769; color: white; }}
</style></head><body>
<h1>学术海报批量测试报告</h1>
<div class="card">
  <p>时间: {report.get('timestamp','')}</p>
  <p>成功/总数: {report.get('success',0)}/{report.get('total',0)}</p>
  <p>综合评分均值: {m.get('overall_avg','-')} | SC: {m.get('semantic_completeness_avg','-')} | LB: {m.get('layout_balance_avg','-')}</p>
  <p>平均迭代: {m.get('iteration_count_avg','-')} | 平均耗时: {m.get('elapsed_sec_avg','-')}s</p>
</div>
<div class="card"><table>
<tr><th>ID</th><th>状态</th><th>Overall</th><th>SC</th><th>LB</th><th>迭代</th><th>耗时</th><th>预览</th></tr>
{rows}
</table></div>
</body></html>"""


def print_summary(report: dict) -> None:
    console.rule("[bold]批量测试报告")
    console.print(f"成功: {report['success']}/{report['total']}")
    if report["failed"]:
        console.print(f"[red]失败: {report['failed']}")
    m = report.get("metrics", {})
    if m:
        console.print(f"SC 均值: {m.get('semantic_completeness_avg')} | LB 均值: {m.get('layout_balance_avg')}")
        console.print(f"综合评分: {m.get('overall_avg')} | 平均迭代: {m.get('iteration_count_avg')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="学术海报批量测试")
    parser.add_argument("--pdf-dir", type=Path, default=Path("samples/papers"))
    parser.add_argument("--manifest", type=Path, help="测试清单 JSON")
    parser.add_argument("--demo-count", type=int, default=0, help="演示模式数量")
    parser.add_argument("--output", type=Path, default=Path("outputs/batch"))
    parser.add_argument("--resume", action="store_true", help="跳过已完成项")
    parser.add_argument("--threshold", type=float, default=None)
    args = parser.parse_args()

    config = Config.load()
    if not config.llm.api_key:
        console.print("[red]未找到 API 密钥")
        return 1

    items = discover_inputs(args.pdf_dir, args.manifest, args.demo_count or 3)
    if not items:
        console.print("[red]未找到测试项")
        return 1

    run_batch_test(items, args.output, resume=args.resume, threshold=args.threshold)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
