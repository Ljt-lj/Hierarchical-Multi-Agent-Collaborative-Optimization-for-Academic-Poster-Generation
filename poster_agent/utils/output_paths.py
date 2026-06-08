"""海报输出路径：按论文名称组织目录."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path


def slugify_paper_title(title: str, *, max_len: int = 60) -> str:
    text = " ".join((title or "").split())
    text = re.sub(r"[^\w\s-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s_-]+", "_", text).strip("_")
    if not text:
        return "paper"
    return text[:max_len].rstrip("_")


def paper_slug_from_source(*, title: str = "", pdf_path: Path | None = None) -> str:
    if title.strip():
        return slugify_paper_title(title)
    if pdf_path is not None:
        return slugify_paper_title(pdf_path.stem)
    return "paper"


def resolve_paper_output_dir(base_output: Path, paper_slug: str) -> Path:
    return base_output / paper_slug


def default_output_name(paper_slug: str, *, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{paper_slug}"


def is_auto_output_name(name: str) -> bool:
    return name in ("", "poster", "auto")


def configure_paper_output(
    config,
    *,
    paper_slug: str,
    output_name: str,
) -> tuple[Path, str]:
    """将输出目录设为 output_base_dir/{paper_slug}/，必要时自动生成 output_name."""
    base = config.output_base_dir.resolve()
    current = config.output_dir.resolve()
    default_dir = resolve_paper_output_dir(base, paper_slug)

    if current.name == paper_slug:
        paper_dir = current
    elif current in (base, default_dir.resolve()):
        paper_dir = default_dir
    else:
        paper_dir = default_dir

    paper_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir = paper_dir
    if is_auto_output_name(output_name):
        output_name = default_output_name(paper_slug)
    return paper_dir, output_name

