"""图像适配：contain / cover 智能选择."""

from __future__ import annotations

from pathlib import Path

from PIL import Image


def _pixel_value(px_val, *, threshold: int = 248) -> int:
    """兼容 Pillow 返回 int 或 (R,G,B) / (R,G,B,A) 像素."""
    if isinstance(px_val, tuple):
        return min(px_val[:3])
    return int(px_val)


def trim_content_margins(img: Image.Image, *, threshold: int = 248) -> Image.Image:
    """裁剪生成图四周近白边，提高有效内容占比."""
    rgb = img.convert("RGB")
    px = rgb.load()
    w, h = rgb.size
    if w <= 2 or h <= 2:
        return rgb

    def row_has_content(y: int) -> bool:
        return any(_pixel_value(px[x, y]) < threshold for x in range(w))

    def col_has_content(x: int) -> bool:
        return any(_pixel_value(px[x, y]) < threshold for y in range(h))

    top = next((y for y in range(h) if row_has_content(y)), 0)
    bottom = next((y for y in range(h - 1, -1, -1) if row_has_content(y)), h - 1)
    left = next((x for x in range(w) if col_has_content(x)), 0)
    right = next((x for x in range(w - 1, -1, -1) if col_has_content(x)), w - 1)
    pad = 4
    left = max(0, left - pad)
    top = max(0, top - pad)
    right = min(w - 1, right + pad)
    bottom = min(h - 1, bottom + pad)
    if right <= left or bottom <= top:
        return rgb
    return rgb.crop((left, top, right + 1, bottom + 1))


def natural_fit_height(img: Image.Image, target_width: int, *, is_generated: bool = False) -> int:
    """按宽度等比缩放后的自然显示高度."""
    if is_generated:
        img = trim_content_margins(img)
    sw, sh = img.size
    if sw <= 0 or target_width <= 0:
        return max(1, int(target_width * 0.35))
    return max(1, int(sh * target_width / sw))


def smart_fit_image(img: Image.Image, tw: int, th: int, *, is_generated: bool = False) -> Image.Image:
    sw, sh = img.size
    if sw <= 0 or sh <= 0 or tw <= 0 or th <= 0:
        return Image.new("RGB", (max(tw, 1), max(th, 1)), (252, 253, 255))
    if is_generated:
        img = trim_content_margins(img)
        sw, sh = img.size
        nh = max(1, int(sh * tw / sw))
        if nh <= th:
            fitted = img.resize((tw, nh), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (tw, nh), (252, 253, 255))
            canvas.paste(fitted, (0, 0))
            return canvas
        return _contain(img, tw, th)
    aspect = sw / sh
    target = tw / th
    ratio_gap = aspect / target if target > 0 else aspect
    extreme = ratio_gap > 1.65 or ratio_gap < 0.58
    if extreme:
        return _cover(img, tw, th)
    return _contain(img, tw, th)


def is_generated_visual(path: str) -> bool:
    p = path.replace("\\", "/").lower()
    return "/visuals/" in p or "_vis" in Path(path).stem.lower()


def _contain(img: Image.Image, tw: int, th: int) -> Image.Image:
    sw, sh = img.size
    scale = min(tw / sw, th / sh)
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (tw, th), (252, 253, 255))
    canvas.paste(img, ((tw - nw) // 2, (th - nh) // 2))
    return canvas


def _cover(img: Image.Image, tw: int, th: int) -> Image.Image:
    sw, sh = img.size
    scale = max(tw / sw, th / sh)
    nw, nh = int(sw * scale), int(sh * scale)
    img = img.resize((nw, nh), Image.Resampling.LANCZOS)
    left = max(0, (nw - tw) // 2)
    top = max(0, (nh - th) // 2)
    return img.crop((left, top, left + tw, top + th))
