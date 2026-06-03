"""根据 VisualSpec 生成图表、流程图、统计卡等图像资产."""

from __future__ import annotations

import io
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

from poster_agent.models.visuals import VisualSpec
from poster_agent.render.text_utils import format_body, smart_title

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

PALETTE = ["#3463AA", "#E8637B", "#F5A623", "#50C878", "#9B59B6", "#1ABC9C"]
CARD_RADIUS = 4
FLOW_RADIUS = 4
STEP_ICONS = ["doc", "gear", "chart", "flag", "star"]


class VisualGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir / "visuals"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all(self, specs: list[VisualSpec], prefix: str) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        for i, spec in enumerate(specs):
            path = self.generate(spec, f"{prefix}_vis{i}")
            if path:
                paths[spec.section_title.lower()] = path
                spec.image_path = str(path)
        return paths

    def generate(self, spec: VisualSpec, name: str) -> Path | None:
        generators = {
            "bar_chart": self._bar_chart,
            "line_chart": self._line_chart,
            "pie_chart": self._pie_chart,
            "flow_diagram": self._flow_diagram,
            "architecture": self._architecture,
            "stat_cards": self._stat_cards,
            "figure": self._figure,
        }
        fn = generators.get(spec.type)
        if not fn:
            return None
        return fn(spec, name)

    def _bar_chart(self, spec: VisualSpec, name: str) -> Path:
        raw_labels = [str(x) for x in spec.data.get("labels", ["A", "B", "C"])]
        values = spec.data.get("values", [70, 85, 92])
        raw_labels, values = _align_label_values(raw_labels, values)
        labels = [_short_label(smart_title(x), 14) for x in raw_labels]
        fig_w = max(4.2, min(len(labels) * 1.05, 7.5))
        fig, ax = plt.subplots(figsize=(fig_w, 3.0), dpi=120)
        colors = PALETTE[: len(values)]
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.2)
        title = smart_title(spec.title or "Results")
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylabel("Score", fontsize=10)
        ax.tick_params(labelsize=8)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=18, ha="right")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f"{val:.0f}" if isinstance(val, float) else str(val),
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )
        fig.tight_layout()
        return self._save_fig(fig, name)

    def _line_chart(self, spec: VisualSpec, name: str) -> Path:
        labels = [str(x) for x in spec.data.get("labels", ["R1", "R2", "R3", "R4"])]
        values = spec.data.get("values", [0.72, 0.78, 0.85, 0.90])
        labels, values = _align_label_values(labels, values)
        fig, ax = plt.subplots(figsize=(5, 3.2), dpi=120)
        ax.plot(labels, values, marker="o", color=PALETTE[0], linewidth=2.5, markersize=7)
        ax.fill_between(range(len(values)), values, alpha=0.12, color=PALETTE[0])
        ax.set_title(smart_title(spec.title or "Trend"), fontsize=12, fontweight="bold", pad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=9)
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        return self._save_fig(fig, name)

    def _pie_chart(self, spec: VisualSpec, name: str) -> Path:
        labels = [smart_title(str(x)) for x in spec.data.get("labels", ["A", "B", "C"])]
        values = spec.data.get("values", [30, 45, 25])
        labels, values = _align_label_values(labels, values)
        fig, ax = plt.subplots(figsize=(4.5, 3.2), dpi=120)
        ax.pie(
            values,
            labels=labels,
            autopct="%1.0f%%",
            colors=PALETTE[: len(values)],
            startangle=90,
            textprops={"fontsize": 9},
        )
        ax.set_title(smart_title(spec.title or "Distribution"), fontsize=12, fontweight="bold")
        fig.tight_layout()
        return self._save_fig(fig, name)

    def _flow_diagram(self, spec: VisualSpec, name: str) -> Path:
        steps = [format_body(str(s)) for s in spec.data.get("steps", ["Step 1", "Step 2", "Step 3"])][:5]
        title = smart_title(spec.title or "Flow")
        if any(k in title.lower() for k in ("architecture", "model", "pipeline", "framework")):
            return self._flow_diagram_vertical(steps, title, name)
        if any(k in title.lower() for k in ("conclusion", "takeaway", "future", "结论", "总结")):
            return self._flow_diagram_vertical(steps, title, name)
        return self._flow_diagram_horizontal(steps, title, name)

    def _flow_diagram_horizontal(self, steps: list[str], title: str, name: str) -> Path:
        n = len(steps)
        font_title = _load_font(14, bold=True)
        font_sm = _load_font(11)

        if n <= 3:
            w, h = 540, 200
            img = Image.new("RGBA", (w, h), (248, 250, 252, 255))
            draw = ImageDraw.Draw(img)
            tw = _text_w(title, font_title)
            draw.text(((w - tw) // 2, 12), title, fill=(35, 45, 60), font=font_title)
            box_w = min(150, (w - 50 - (n - 1) * 24) // max(n, 1))
            gap = 24
            start_x = (w - (n * box_w + (n - 1) * gap)) // 2
            y = 44
            for i, step in enumerate(steps):
                x = start_x + i * (box_w + gap)
                lines = _wrap_step_text(step, font_sm, box_w - 44)
                box_h = max(56, 20 + len(lines) * 14)
                color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
                _draw_shadow_box(draw, x, y, box_w, box_h, color, radius=FLOW_RADIUS)
                _draw_step_icon(draw, x + 8, y + 8, STEP_ICONS[i % len(STEP_ICONS)], color)
                _draw_lines_in_box(draw, x + 36, y + 10, box_w - 44, lines, font_sm)
                if i < n - 1:
                    ax = x + box_w + 4
                    mid_y = y + box_h // 2
                    draw.line([(ax, mid_y), (ax + gap - 8, mid_y)], fill=(130, 140, 155), width=2)
        else:
            w, h = 520, 40 + n * 52
            img = Image.new("RGBA", (w, h), (248, 250, 252, 255))
            draw = ImageDraw.Draw(img)
            draw.text((16, 12), title, fill=(35, 45, 60), font=font_title)
            y = 40
            box_w = w - 32
            for i, step in enumerate(steps):
                lines = _wrap_step_text(step, font_sm, box_w - 44)
                box_h = max(44, 16 + len(lines) * 14)
                color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
                _draw_shadow_box(draw, 16, y, box_w, box_h, color, radius=FLOW_RADIUS)
                _draw_step_icon(draw, 24, y + 8, STEP_ICONS[i % len(STEP_ICONS)], color)
                _draw_lines_in_box(draw, 52, y + 10, box_w - 52, lines, font_sm)
                y += box_h + 10

        out = self.output_dir / f"{name}.png"
        img.convert("RGB").save(out)
        return out

    def _flow_diagram_vertical(self, steps: list[str], title: str, name: str) -> Path:
        """纵向流程图，适合 Architecture 等竖向排版区块."""
        font_title = _load_font(15, bold=True)
        font_sm = _load_font(12)
        w = 480
        box_w = w - 32
        y = 36
        rows: list[tuple[int, int]] = []
        for step in steps:
            lines = _wrap_step_text(step, font_sm, box_w - 48)
            box_h = max(52, 18 + len(lines) * 15)
            rows.append((box_h, len(lines)))
            y += box_h + 18
        h = y + 8
        img = Image.new("RGBA", (w, h), (248, 250, 252, 255))
        draw = ImageDraw.Draw(img)
        tw = _text_w(title, font_title)
        draw.text(((w - tw) // 2, 10), title, fill=(35, 45, 60), font=font_title)
        y = 36
        for i, step in enumerate(steps):
            lines = _wrap_step_text(step, font_sm, box_w - 48)
            box_h = max(52, 18 + len(lines) * 15)
            color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            _draw_shadow_box(draw, 16, y, box_w, box_h, color, radius=FLOW_RADIUS)
            _draw_step_icon(draw, 24, y + 10, STEP_ICONS[i % len(STEP_ICONS)], color)
            _draw_lines_in_box(draw, 54, y + 12, box_w - 58, lines, font_sm)
            if i < len(steps) - 1:
                cx = w // 2
                draw.line([(cx, y + box_h + 2), (cx, y + box_h + 16)], fill=(130, 140, 155), width=2)
            y += box_h + 18

        out = self.output_dir / f"{name}.png"
        img.convert("RGB").save(out)
        return out

    def _architecture(self, spec: VisualSpec, name: str) -> Path:
        steps = spec.data.get("steps", ["Input", "Process", "Output"])
        return self._flow_diagram(
            VisualSpec(type="flow_diagram", title=spec.title or "Architecture", data={"steps": steps}),
            name,
        )

    def _stat_cards(self, spec: VisualSpec, name: str) -> Path:
        cards = spec.data.get("cards", [
            {"label": "Metric A", "value": "92", "unit": "%"},
            {"label": "Metric B", "value": "4.5", "unit": "/5"},
        ])
        n = len(cards)
        w, h = 540, 175
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        font_val = _load_font(38, bold=True)
        font_unit = _load_font(16)
        font_lbl = _load_font(14)
        pad = 16
        card_w = (w - pad * (n + 1)) // n

        for i, card in enumerate(cards):
            x = pad + i * (card_w + pad)
            y0, y1 = 24, h - 24
            color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            _draw_shadow_box(draw, x, y0, card_w, y1 - y0, color, radius=CARD_RADIUS, fill=(255, 255, 255))
            draw.rectangle([x, y0, x + card_w, y0 + 6], fill=color)
            val = str(card.get("value", ""))[:8]
            unit = str(card.get("unit", ""))[:6]
            label = smart_title(str(card.get("label", "")))
            label_lines = textwrap.wrap(label, width=max(8, card_w // 10))[:2]
            draw.text((x + 10, y0 + 18), val, fill=color, font=font_val)
            if unit:
                vw = _text_w(val, font_val)
                draw.text((x + 10 + vw + 2, y0 + 32), unit, fill=(110, 118, 130), font=font_unit)
            ly = y1 - 28
            for line in label_lines:
                draw.text((x + 10, ly), line, fill=(60, 68, 80), font=font_lbl)
                ly += 16

        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _figure(self, spec: VisualSpec, name: str) -> Path | None:
        from poster_agent.render.image_fit import smart_fit_image

        src = spec.image_path or spec.data.get("path")
        if not src or not Path(src).exists():
            return None
        img = Image.open(src).convert("RGB")
        fitted = smart_fit_image(img, 540, 380, is_generated=False)
        out = self.output_dir / f"{name}.png"
        fitted.save(out)
        return out

    def _save_fig(self, fig, name: str) -> Path:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="white")
        plt.close(fig)
        buf.seek(0)
        out = self.output_dir / f"{name}.png"
        Image.open(buf).convert("RGB").save(out)
        return out


def _draw_shadow_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    w: int,
    h: int,
    accent: tuple[int, int, int],
    *,
    radius: int = 4,
    fill: tuple[int, int, int] = (255, 255, 255),
) -> None:
    shadow = (190, 198, 210)
    draw.rounded_rectangle([x + 2, y + 3, x + w + 2, y + h + 3], radius=radius, fill=shadow)
    draw.rounded_rectangle([x, y, x + w, y + h], radius=radius, fill=fill, outline=accent, width=2)


def _draw_step_icon(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    icon: str,
    color: tuple[int, int, int],
) -> None:
    s = 22
    draw.rounded_rectangle([x, y, x + s, y + s], radius=3, fill=color)
    cx, cy = x + s // 2, y + s // 2
    white = (255, 255, 255)
    if icon == "chart":
        draw.rectangle([cx - 5, cy, cx - 2, cy + 5], fill=white)
        draw.rectangle([cx - 1, cy - 3, cx + 2, cy + 5], fill=white)
        draw.rectangle([cx + 4, cy + 2, cx + 7, cy + 5], fill=white)
    elif icon == "gear":
        draw.ellipse([cx - 5, cy - 5, cx + 5, cy + 5], outline=white, width=1)
        draw.ellipse([cx - 2, cy - 2, cx + 2, cy + 2], fill=white)
    elif icon == "flag":
        draw.line([(cx - 3, cy - 6), (cx - 3, cy + 6)], fill=white, width=2)
        draw.polygon([(cx - 3, cy - 6), (cx + 6, cy - 3), (cx - 3, cy)], fill=white)
    elif icon == "star":
        draw.polygon([(cx, cy - 5), (cx + 2, cy), (cx + 5, cy + 1), (cx + 3, cy + 4), (cx + 4, cy + 7), (cx, cy + 5), (cx - 4, cy + 7), (cx - 3, cy + 4), (cx - 5, cy + 1), (cx - 2, cy)], fill=white)
    else:
        draw.rectangle([cx - 5, cy - 4, cx + 5, cy + 5], outline=white, width=1)


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["msyhbd.ttc", "msyh.ttc", "simhei.ttf", "arialbd.ttf" if bold else "arial.ttf"]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_w(text: str, font) -> int:
    try:
        return int(font.getlength(text))
    except Exception:
        return len(text) * 12


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _short_label(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 2] + ".."


def _align_label_values(
    labels: list,
    values: list,
    *,
    max_items: int = 8,
) -> tuple[list, list]:
    labels = list(labels or [])
    values = list(values or [])
    if not labels and not values:
        return ["A", "B", "C"], [70, 85, 92]
    if not labels:
        labels = [f"M{i + 1}" for i in range(len(values))]
    if not values:
        values = [70 + i * 5 for i in range(len(labels))]
    n = min(len(labels), len(values), max_items)
    return labels[:n], values[:n]


def _wrap_step_text(text: str, font, max_w: int) -> list[str]:
    if max_w < 20:
        return [_short_label(text, 12)]
    avg_char = max(6, max_w // 8)
    lines = textwrap.wrap(text, width=avg_char)
    return lines[:3] if lines else [_short_label(text, 20)]


def _draw_lines_in_box(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    max_w: int,
    lines: list[str],
    font,
) -> None:
    cy = y
    for line in lines:
        if _text_w(line, font) > max_w:
            while line and _text_w(line + "…", font) > max_w:
                line = line[:-1]
            line = (line + "…") if line else ""
        draw.text((x, cy), line, fill=(30, 35, 45), font=font)
        cy += 14
