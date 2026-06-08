"""论文插图智能裁剪：按空白带切分子 panel，生成海报可读性子图."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from poster_agent.agents.figure_curator import FigureAsset, classify_figure
from poster_agent.render.image_fit import trim_content_margins, _pixel_value

# 各 Fig 在 focus 图内的兜底相对裁剪（top/bottom/left/right 为保留比例）
_FALLBACK_CROPS: dict[tuple[int, str], dict[str, float]] = {
    (1, "intro"): {"top": 0.0, "bottom": 0.50, "left": 0.0, "right": 0.0},
    (2, "benchmark"): {"top": 0.0, "bottom": 0.36, "left": 0.0, "right": 0.0},
    (3, "structure"): {"top": 0.18, "bottom": 0.42, "left": 0.0, "right": 0.0},
}

_ROLE_SUFFIX = {
    "intro": "crop_intro",
    "benchmark": "crop_benchmark",
    "structure": "crop_structure",
}


def derive_poster_crops(
    focus_asset: FigureAsset,
    output_dir: Path,
    *,
    captions: dict[int, str] | None = None,
) -> list[FigureAsset]:
    """从 focus 图生成 intro / benchmark / structure 子图并写入 catalog 资产."""
    fig_num = focus_asset.fig_num
    if fig_num <= 0:
        return []
    roles = _roles_for_figure(fig_num)
    if not roles:
        return []

    try:
        img = Image.open(focus_asset.path).convert("RGB")
    except Exception:
        return []

    caps = captions or {}
    cap = focus_asset.caption_hint or caps.get(fig_num, "")
    out: list[FigureAsset] = []

    for role in roles:
        cropped = crop_for_poster_role(img, fig_num, role)
        if cropped.size[0] < 80 or cropped.size[1] < 60:
            continue
        suffix = _ROLE_SUFFIX.get(role, f"crop_{role}")
        out_path = output_dir / f"fig{fig_num}_{suffix}.png"
        cropped.save(out_path, optimize=True)
        w, h = cropped.size
        asset = FigureAsset(
            str(out_path),
            focus_asset.page,
            w,
            h,
            float(w * h) * 22.0,
            caption_hint=cap[:200],
            fig_num=fig_num,
            poster_role=role,
        )
        asset.kind = classify_figure(asset)
        out.append(asset)
    return out


def _roles_for_figure(fig_num: int) -> list[str]:
    if fig_num == 1:
        return ["intro"]
    if fig_num == 2:
        return ["benchmark"]
    if fig_num == 3:
        return ["structure"]
    return []


def crop_for_poster_role(img: Image.Image, fig_num: int, role: str) -> Image.Image:
    """按空白带切分或兜底比例裁剪，并去除四周近白边."""
    trimmed = trim_content_margins(img, threshold=244)
    bands = split_horizontal_bands(trimmed, min_gap=14, white_ratio=0.965, min_band_h=70)
    picked = _pick_band_for_role(bands, trimmed, fig_num, role)
    return trim_content_margins(picked, threshold=244)


def split_horizontal_bands(
    img: Image.Image,
    *,
    min_gap: int = 14,
    white_ratio: float = 0.965,
    min_band_h: int = 60,
    threshold: int = 248,
) -> list[Image.Image]:
    """在水平空白带处切分图像为若干条带."""
    rgb = img.convert("RGB")
    w, h = rgb.size
    if h < min_band_h * 2:
        return [rgb]

    px = rgb.load()
    row_white = [
        sum(1 for x in range(w) if _pixel_value(px[x, y], threshold=threshold) >= threshold) / w
        for y in range(h)
    ]

    gaps: list[tuple[int, int]] = []
    y = 0
    while y < h:
        if row_white[y] >= white_ratio:
            gap_start = y
            while y < h and row_white[y] >= white_ratio:
                y += 1
            if y - gap_start >= min_gap:
                gaps.append((gap_start, y))
        else:
            y += 1

    if not gaps:
        return [rgb]

    cuts = [0]
    for g0, g1 in gaps:
        cuts.append((g0 + g1) // 2)
    cuts.append(h)

    bands: list[Image.Image] = []
    for i in range(len(cuts) - 1):
        y0, y1 = cuts[i], cuts[i + 1]
        if y1 - y0 < min_band_h:
            continue
        band = rgb.crop((0, y0, w, y1))
        # 跳过几乎全白的条带
        if _band_has_content(band, threshold=threshold):
            bands.append(band)
    return bands or [rgb]


def _band_has_content(img: Image.Image, *, threshold: int = 248) -> bool:
    rgb = img.convert("RGB")
    w, h = rgb.size
    if w <= 0 or h <= 0:
        return False
    px = rgb.load()
    content = sum(
        1 for y in range(h) for x in range(w) if _pixel_value(px[x, y], threshold=threshold) < threshold
    )
    return content > w * h * 0.02


def _pick_band_for_role(
    bands: list[Image.Image],
    full: Image.Image,
    fig_num: int,
    role: str,
) -> Image.Image:
    n = len(bands)
    if fig_num == 1 and role == "intro":
        return bands[0] if n >= 1 else _relative_crop(full, _FALLBACK_CROPS[(1, "intro")])

    if fig_num == 2 and role == "benchmark":
        # 去掉最底部结构可视化行，保留上方统计图
        if n >= 2:
            return _merge_vertical(bands[:-1])
        return _relative_crop(full, _FALLBACK_CROPS[(2, "benchmark")])

    if fig_num == 3 and role == "structure":
        # 优先取含结构示意图的中段（通常为第 2 条带）
        if n >= 3:
            return bands[1]
        if n == 2:
            return bands[1]
        return _relative_crop(full, _FALLBACK_CROPS[(3, "structure")])

    profile = _FALLBACK_CROPS.get((fig_num, role))
    return _relative_crop(full, profile) if profile else full


def _merge_vertical(bands: list[Image.Image]) -> Image.Image:
    if len(bands) == 1:
        return bands[0]
    w = max(b.width for b in bands)
    total_h = sum(b.height for b in bands)
    out = Image.new("RGB", (w, total_h), (255, 255, 255))
    y = 0
    for band in bands:
        x_off = (w - band.width) // 2
        out.paste(band, (x_off, y))
        y += band.height
    return out


def _relative_crop(img: Image.Image, profile: dict[str, float]) -> Image.Image:
    w, h = img.size
    top = int(h * profile.get("top", 0))
    bottom = int(h * profile.get("bottom", 0))
    left = int(w * profile.get("left", 0))
    right = int(w * profile.get("right", 0))
    y0, y1 = top, max(top + 40, h - bottom)
    x0, x1 = left, max(left + 40, w - right)
    if y1 <= y0 or x1 <= x0:
        return img
    return img.crop((x0, y0, x1, y1))


def is_poster_crop_path(path: str) -> bool:
    name = Path(path).name.lower()
    return "_crop_" in name
