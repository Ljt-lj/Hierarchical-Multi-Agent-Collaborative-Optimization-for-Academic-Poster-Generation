#!/usr/bin/env python3
"""下载真实英文论文并批量生成学术海报.

每篇论文独立文件夹，输出文件命名为「标题_生成时间」。

用法:
  python experiments/real_paper_test.py
  python experiments/real_paper_test.py --skip-download
  python experiments/real_paper_test.py --count 3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from poster_agent.config import Config
from poster_agent.pipeline import PosterPipeline

console = Console()

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# 五篇经典英文论文（arXiv）
DEFAULT_PAPERS = [
    {"arxiv_id": "1706.03762", "fallback_title": "Attention Is All You Need"},
    {"arxiv_id": "1810.04805", "fallback_title": "BERT Pre-training of Deep Bidirectional Transformers"},
    {"arxiv_id": "2005.14165", "fallback_title": "Language Models are Few-Shot Learners"},
    {"arxiv_id": "2103.00020", "fallback_title": "Learning Transferable Visual Models From Natural Language Supervision"},
    {"arxiv_id": "2010.11929", "fallback_title": "An Image is Worth 16x16 Words Transformers for Image Recognition at Scale"},
]


def slugify(text: str, max_len: int = 55) -> str:
    text = smart_ascii_title(text)
    s = re.sub(r"[^\w\s-]", "", text)
    s = re.sub(r"[\s_-]+", "_", s).strip("_")
    return (s[:max_len] if s else "paper")


def smart_ascii_title(text: str) -> str:
    return " ".join(text.split())


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "PosterAgentTest/1.0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        xml_data = resp.read()
    root = ET.fromstring(xml_data)
    entry = root.find("atom:entry", ATOM_NS)
    if entry is None:
        return {"arxiv_id": arxiv_id, "title": arxiv_id, "authors": []}
    title_el = entry.find("atom:title", ATOM_NS)
    title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else arxiv_id
    authors = [
        a.find("atom:name", ATOM_NS).text.strip()
        for a in entry.findall("atom:author", ATOM_NS)
        if a.find("atom:name", ATOM_NS) is not None and a.find("atom:name", ATOM_NS).text
    ]
    return {"arxiv_id": arxiv_id, "title": title, "authors": authors[:5]}


def download_pdf(arxiv_id: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    console.print(f"  下载 PDF: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "PosterAgentTest/1.0"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = resp.read()
    if len(data) < 10_000:
        raise RuntimeError(f"PDF 过小，可能下载失败: {arxiv_id}")
    dest.write_bytes(data)
    return dest


def prepare_paper(
    spec: dict,
    papers_dir: Path,
) -> dict:
    arxiv_id = spec["arxiv_id"]
    try:
        meta = fetch_arxiv_metadata(arxiv_id)
    except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as e:
        console.print(f"[yellow]元数据获取失败 {arxiv_id}: {e}，使用备用标题")
        meta = {
            "arxiv_id": arxiv_id,
            "title": spec.get("fallback_title", arxiv_id),
            "authors": [],
        }

    title = meta["title"] or spec.get("fallback_title", arxiv_id)
    slug = slugify(title)
    paper_dir = papers_dir / slug
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / f"{arxiv_id}.pdf"
    download_pdf(arxiv_id, pdf_path)

    return {
        **meta,
        "title": title,
        "slug": slug,
        "pdf_path": str(pdf_path),
        "paper_dir": str(paper_dir),
    }


def run_poster_for_paper(
    paper: dict,
    batch_root: Path,
    config: Config,
) -> dict:
    slug = paper["slug"]
    title = paper["title"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_name = f"{ts}_{slug}"

    # 按论文组织: outputs/real_paper_test/{slug}/
    paper_out = batch_root / slug
    paper_out.mkdir(parents=True, exist_ok=True)

    # 保存源 PDF 副本
    src_pdf = Path(paper["pdf_path"])
    dest_pdf = paper_out / "source" / src_pdf.name
    dest_pdf.parent.mkdir(parents=True, exist_ok=True)
    if not dest_pdf.exists():
        shutil.copy2(src_pdf, dest_pdf)

    config.output_dir = paper_out
    pipeline = PosterPipeline(config)

    t0 = time.time()
    result = pipeline.run(pdf_path=src_pdf, output_name=output_name)
    elapsed = round(time.time() - t0, 1)

    record = {
        "arxiv_id": paper["arxiv_id"],
        "title": title,
        "slug": slug,
        "authors": paper.get("authors", []),
        "output_name": output_name,
        "output_dir": str(paper_out),
        "png": str(result.png_path),
        "pptx": str(result.pptx_path),
        "result_png": str(result.result_png),
        "result_pptx": str(result.result_pptx),
        "best_iteration": result.best_iteration,
        "elapsed_sec": elapsed,
        "overall_score": round(result.final_score.overall, 3),
        "best_score": round(
            max((h.get("overall", 0) for h in result.iteration_history), default=result.final_score.overall),
            3,
        ),
        "iterations": len(result.iteration_history),
        "generated_at": ts,
    }

    meta_path = paper_out / f"{output_name}_meta.json"
    meta_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _find_cached_pdf(papers_dir: Path, arxiv_id: str, slug: str) -> Path | None:
    candidates = [
        papers_dir / slug / f"{arxiv_id}.pdf",
        papers_dir / f"{arxiv_id}.pdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    if papers_dir.exists():
        for path in papers_dir.rglob(f"{arxiv_id}.pdf"):
            return path
    return None


def run_test(
    *,
    count: int = 5,
    skip_download: bool = False,
    output_root: Path | None = None,
    arxiv_id: str | None = None,
) -> dict:
    config = Config.load()
    if not config.llm.api_key:
        raise RuntimeError("未找到 API 密钥，请配置 api_key.txt")

    batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_root = output_root or ROOT / "outputs" / "real_paper_test" / batch_ts
    papers_dir = ROOT / "samples" / "papers" / "real"
    batch_root.mkdir(parents=True, exist_ok=True)

    specs = DEFAULT_PAPERS[:count]
    if arxiv_id:
        specs = [s for s in DEFAULT_PAPERS if s["arxiv_id"] == arxiv_id]
        if not specs:
            specs = [{"arxiv_id": arxiv_id, "fallback_title": arxiv_id}]
    console.rule("[bold]真实论文测试")
    console.print(f"论文数量: {len(specs)}")
    console.print(f"输出根目录: {batch_root}")

    papers: list[dict] = []
    if not skip_download:
        console.rule("下载论文")
        for spec in specs:
            try:
                paper = prepare_paper(spec, papers_dir)
                papers.append(paper)
                console.print(f"[green]OK[/green] {paper['title'][:70]}...")
            except Exception as e:
                console.print(f"[red]FAIL[/red] {spec['arxiv_id']}: {e}")
    else:
        for spec in specs:
            try:
                meta = fetch_arxiv_metadata(spec["arxiv_id"])
            except (urllib.error.URLError, TimeoutError, urllib.error.HTTPError) as e:
                console.print(f"[yellow]元数据获取失败 {spec['arxiv_id']}: {e}，使用备用标题")
                meta = {
                    "arxiv_id": spec["arxiv_id"],
                    "title": spec.get("fallback_title", spec["arxiv_id"]),
                    "authors": [],
                }
            title = meta["title"] or spec.get("fallback_title", spec["arxiv_id"])
            slug = slugify(title)
            pdf_path = _find_cached_pdf(papers_dir, spec["arxiv_id"], slug)
            if pdf_path is None:
                console.print(f"[red]缺少 PDF: {papers_dir / slug / spec['arxiv_id']}.pdf，请先运行不带 --skip-download")
                continue
            papers.append({**meta, "title": title, "slug": slug, "pdf_path": str(pdf_path)})

    if not papers:
        raise RuntimeError("没有可用的论文")

    console.rule("生成海报")
    results: list[dict] = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("生成中...", total=len(papers))
        for paper in papers:
            progress.update(task, description=f"处理: {paper['slug'][:40]}")
            try:
                record = run_poster_for_paper(paper, batch_root, config)
                results.append(record)
                console.print(
                    f"[green]完成[/green] {record['title'][:50]}... "
                    f"score={record['overall_score']} best={record.get('best_score')} ({record['elapsed_sec']}s)"
                )
                console.print(f"  结果: {record.get('result_png')}")
            except Exception as e:
                console.print(f"[red]失败[/red] {paper.get('title', '')}: {e}")
                results.append({"title": paper.get("title"), "error": str(e), "slug": paper.get("slug")})
            progress.advance(task)

    report = {
        "batch_timestamp": batch_ts,
        "output_root": str(batch_root),
        "paper_count": len(results),
        "success": sum(1 for r in results if "error" not in r),
        "papers": results,
    }
    report_path = batch_root / "batch_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    console.rule("[bold]测试完成")
    console.print(f"成功: {report['success']}/{report['paper_count']}")
    console.print(f"报告: {report_path}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="下载真实英文论文并生成学术海报")
    parser.add_argument("--count", type=int, default=5, help="论文数量（默认 5）")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载，使用已缓存 PDF")
    parser.add_argument("--output", type=Path, default=None, help="输出根目录")
    parser.add_argument("--arxiv-id", type=str, default=None, help="仅处理指定 arXiv ID（如 1706.03762）")
    args = parser.parse_args()

    try:
        run_test(
            count=args.count,
            skip_download=args.skip_download,
            output_root=args.output,
            arxiv_id=args.arxiv_id,
        )
        return 0
    except Exception as e:
        console.print(f"[red]错误:[/red] {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
