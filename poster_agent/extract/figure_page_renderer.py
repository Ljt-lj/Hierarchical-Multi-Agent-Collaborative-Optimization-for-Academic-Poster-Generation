"""从 PDF 页面渲染完整论文插图（多 panel 流程图 / 数据图），避免嵌入碎片."""

from __future__ import annotations

import io
import re
from pathlib import Path

import fitz
from PIL import Image

from poster_agent.agents.figure_curator import FigureAsset, classify_figure


def extract_figure_captions_from_pdf(doc: fitz.Document) -> dict[int, str]:
    """从 PDF 各页直接读取 Fig. N | caption 格式."""
    pattern = re.compile(r"Fig\.?\s*(\d+)\s*[|:\-–—]\s*([^\n]{15,240})", re.I)
    captions: dict[int, str] = {}
    for page in doc:
        for m in pattern.finditer(page.get_text("text")):
            num = int(m.group(1))
            cap = re.sub(r"\s+", " ", m.group(2)).strip()
            if num not in captions or len(cap) > len(captions[num]):
                captions[num] = cap
    return captions


def detect_figure_caption_pages(doc: fitz.Document) -> dict[int, int]:
    """检测各 Fig 的 caption 所在页（0-indexed）."""
    pattern = re.compile(r"Fig\.?\s*(\d+)\s*[|:\-–—]", re.I)
    mapping: dict[int, int] = {}
    for page_idx, page in enumerate(doc):
        for m in pattern.finditer(page.get_text("text")):
            num = int(m.group(1))
            if num not in mapping:
                mapping[num] = page_idx
    return mapping


def figure_page_spans(caption_pages: dict[int, int], page_count: int, *, max_main: int = 6) -> dict[int, list[int]]:
    """Fig N 可能跨页：caption 页到下一 Fig caption 页之间的页面均归属 Fig N."""
    spans: dict[int, list[int]] = {}
    items = [(n, p) for n, p in sorted(caption_pages.items()) if n <= max_main]
    for i, (num, start) in enumerate(items):
        end = items[i + 1][1] if i + 1 < len(items) else min(start + 2, page_count)
        pages = list(range(start, min(end, page_count)))
        spans[num] = pages[:2]
    return spans


def _clip_rect(page: fitz.Page, *, top: float = 0.10, bottom: float = 0.24, margin: float = 0.04) -> fitz.Rect:
    rect = page.rect
    return fitz.Rect(
        rect.x0 + rect.width * margin,
        rect.y0 + rect.height * top,
        rect.x1 - rect.width * margin,
        rect.y1 - rect.height * bottom,
    )


def _render_page_region(
    page: fitz.Page,
    *,
    dpi: int = 220,
    top: float = 0.10,
    bottom: float = 0.24,
    margin: float = 0.04,
) -> Image.Image:
    rect = page.rect
    clip = fitz.Rect(
        rect.x0 + rect.width * margin,
        rect.y0 + rect.height * top,
        rect.x1 - rect.width * margin,
        rect.y1 - rect.height * bottom,
    )
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), clip=clip)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


# 海报用：单页聚焦裁剪（去掉页眉/正文/caption 区，只保留核心 panel）
POSTER_FOCUS_CROPS: dict[int, dict[str, float]] = {
    1: {"top": 0.07, "bottom": 0.58, "margin": 0.025},
    2: {"top": 0.07, "bottom": 0.60, "margin": 0.03},
    3: {"top": 0.08, "bottom": 0.32, "margin": 0.035},
}


def render_figure_focus_panels(
    pdf_path: Path,
    output_dir: Path,
    captions: dict[int, str],
    *,
    dpi: int = 240,
    max_figures: int = 3,
) -> list[FigureAsset]:
    """渲染 Fig.1–3 的聚焦 panel（非整页截图），提高海报可读性."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pdf_captions = extract_figure_captions_from_pdf(doc)
    merged_captions = dict(captions)
    for num, cap in pdf_captions.items():
        if num not in merged_captions or len(cap) > len(merged_captions.get(num, "")):
            merged_captions[num] = cap

    caption_pages = detect_figure_caption_pages(doc)
    assets: list[FigureAsset] = []

    for fig_num in sorted(caption_pages.keys()):
        if fig_num > max_figures:
            continue
        crop = POSTER_FOCUS_CROPS.get(fig_num, {"top": 0.10, "bottom": 0.30, "margin": 0.04})
        page_idx = caption_pages[fig_num]
        try:
            image = _render_page_region(
                doc[page_idx],
                dpi=dpi,
                top=crop["top"],
                bottom=crop["bottom"],
                margin=crop["margin"],
            )
        except Exception:
            continue

        out_path = output_dir / f"fig{fig_num}_focus.png"
        image.save(out_path, optimize=True)
        w, h = image.size
        cap = merged_captions.get(fig_num, "")
        asset = FigureAsset(
            str(out_path),
            page_idx + 1,
            w,
            h,
            float(w * h) * 18.0,
            caption_hint=cap[:200],
            fig_num=fig_num,
        )
        asset.kind = classify_figure(asset)
        assets.append(asset)
        from poster_agent.extract.figure_cropper import derive_poster_crops

        assets.extend(derive_poster_crops(asset, output_dir, captions=merged_captions))

    doc.close()
    return assets


def _stitch_vertical(images: list[Image.Image]) -> Image.Image:
    if len(images) == 1:
        return images[0]
    w = max(im.width for im in images)
    total_h = sum(im.height for im in images)
    out = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for im in images:
        x_off = (w - im.width) // 2
        out.paste(im, (x_off, y))
        y += im.height
    return out


def render_figure_composites(
    pdf_path: Path,
    output_dir: Path,
    captions: dict[int, str],
    *,
    dpi: int = 220,
    max_figures: int = 6,
) -> list[FigureAsset]:
    """渲染 Fig.1–N 完整页面区域为 composite PNG，供海报优先使用."""
    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    pdf_captions = extract_figure_captions_from_pdf(doc)
    merged_captions = dict(captions)
    for num, cap in pdf_captions.items():
        if num not in merged_captions or len(cap) > len(merged_captions.get(num, "")):
            merged_captions[num] = cap

    caption_pages = detect_figure_caption_pages(doc)
    if not caption_pages:
        doc.close()
        return []

    spans = figure_page_spans(caption_pages, len(doc), max_main=max_figures)
    assets: list[FigureAsset] = []

    for fig_num in sorted(spans.keys()):
        if fig_num > max_figures:
            continue
        page_indices = spans[fig_num]
        try:
            images = [
                _render_page_region(doc[i], dpi=dpi, top=0.08, bottom=0.16, margin=0.03)
                for i in page_indices
            ]
            composite = _stitch_vertical(images)
        except Exception:
            continue

        out_path = output_dir / f"fig{fig_num}_composite.png"
        composite.save(out_path, optimize=True)
        w, h = composite.size
        cap = merged_captions.get(fig_num, "")
        asset = FigureAsset(
            str(out_path),
            page_indices[0] + 1,
            w,
            h,
            float(w * h) * 8.0,
            caption_hint=cap[:200],
            fig_num=fig_num,
        )
        asset.kind = classify_figure(asset)
        assets.append(asset)

    doc.close()
    assets.sort(key=lambda a: (a.fig_num, -a.score))
    return assets


def merge_composite_and_embedded(
    composites: list[FigureAsset],
    embedded: list[FigureAsset],
) -> list[FigureAsset]:
    """有 composite 时丢弃同 fig_num 的嵌入碎片."""
    covered = {a.fig_num for a in composites if a.fig_num}
    kept_embedded = [a for a in embedded if a.fig_num not in covered or a.fig_num == 0]
    merged = composites + kept_embedded
    merged.sort(key=lambda a: (-a.score, a.page))
    return merged


def merge_poster_figure_assets(
    focus: list[FigureAsset],
    composites: list[FigureAsset],
    embedded: list[FigureAsset],
) -> list[FigureAsset]:
    """聚焦 panel 优先，其次 composite，最后嵌入碎片."""
    covered = {a.fig_num for a in focus if a.fig_num}
    comps = [a for a in composites if a.fig_num not in covered]
    covered |= {a.fig_num for a in comps if a.fig_num}
    emb = [a for a in embedded if a.fig_num not in covered or a.fig_num == 0]
    merged = focus + comps + emb
    merged.sort(key=lambda a: (-a.score, a.page))
    return merged
