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
from poster_agent.render.poster_style import (
    ACCENT_RGB,
    ARROW_GRAY,
    BG_TABLE_ALT,
    CHART_ACCENT,
    CHART_NEUTRAL,
    FLOW_BORDER,
    FLOW_FILL,
    FLOW_TITLE,
    chart_colors,
)
from poster_agent.render.text_utils import format_body, smart_title

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 保留旧名兼容；新图统一用 poster_style
PALETTE = ["#C45C26", "#9A9A9A", "#B5B5B5", "#757575", "#C4A882", "#8E8E8E"]
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
            "logic_pipeline": self._logic_pipeline,
            "architecture": self._architecture,
            "stat_cards": self._stat_cards,
            "bullet_cards": self._bullet_cards,
            "data_table": self._data_table,
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
        fig_w = max(5.5, min(len(labels) * 1.25, 9.0))
        fig, ax = plt.subplots(figsize=(fig_w, 3.4), dpi=140)
        colors = chart_colors(labels)
        bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.0)
        title = smart_title(spec.title or "Results")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10, color="#1C1C1C")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")
        ax.spines["bottom"].set_color("#CCCCCC")
        ax.set_ylabel("Score", fontsize=10, color="#555555")
        ax.tick_params(labelsize=8, colors="#444444")
        ax.set_facecolor("white")
        fig.patch.set_facecolor("white")
        max_val = max(float(v) for v in values) if values else 1
        y_pad = max(max_val * 0.15, 0.05) if max_val < 10 else max_val * 0.15 + 2
        ax.set_ylim(0, max_val + y_pad)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=18, ha="right")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + y_pad * 0.08,
                _format_bar_value(val),
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

    def _logic_pipeline(self, spec: VisualSpec, name: str) -> Path:
        """逻辑流程图：每步含名称、机制说明、数据流（基于 LogicPlan）."""
        raw_steps = spec.data.get("steps") or []
        layout = str(spec.data.get("layout", "")).lower()
        steps: list[dict[str, str]] = []
        for item in raw_steps[:6]:
            if isinstance(item, dict):
                steps.append({
                    "name": str(item.get("name", ""))[:32],
                    "detail": format_body(str(item.get("detail", "")))[:140],
                    "io": str(item.get("io", ""))[:40],
                })
            else:
                text = format_body(str(item))
                if ":" in text:
                    n, d = text.split(":", 1)
                    steps.append({"name": n.strip()[:32], "detail": d.strip()[:140], "io": ""})
                else:
                    steps.append({"name": text[:32], "detail": "", "io": ""})
        if len(steps) < 3:
            steps = [{"name": s["name"], "detail": s.get("detail", ""), "io": ""} for s in _steps_to_layers(
                [format_body(str(s)) for s in raw_steps[:4]]
            )]

        title = smart_title(spec.title or "Method Pipeline")
        if layout == "horizontal":
            return self._logic_pipeline_horizontal(steps, title, name)
        return self._logic_pipeline_vertical(steps, title, name)

    def _logic_pipeline_vertical(self, steps: list[dict[str, str]], title: str, name: str) -> Path:
        """纵向白底黑框流程图（对标参考海报 Methods）."""
        font_title = _load_font(16, bold=True)
        font_name = _load_font(12, bold=True)
        font_detail = _load_font(10)
        w = 560
        box_w = w - 40
        inner = box_w - 20
        y = 36
        row_heights: list[int] = []
        for step in steps:
            detail_lines = _wrap_to_width(step.get("detail", ""), font_detail, inner)[:2]
            h = 14 + 16 + len(detail_lines) * 12 + 10
            row_heights.append(max(h, 48))
            y += max(h, 48) + 20
        h = y + 4
        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        tw = _text_w(title, font_title)
        draw.text(((w - tw) // 2, 8), title, fill=FLOW_TITLE, font=font_title)
        y = 36
        for i, step in enumerate(steps):
            row_h = row_heights[i]
            _draw_academic_box(draw, 20, y, box_w, row_h)
            cy = y + 8
            for line in _wrap_to_width(step["name"], font_name, inner)[:1]:
                draw.text((32, cy), line, fill=FLOW_TITLE, font=font_name)
                cy += 15
            for line in _wrap_to_width(step.get("detail", ""), font_detail, inner)[:2]:
                draw.text((32, cy), line, fill=(80, 80, 80), font=font_detail)
                cy += 12
            if i < len(steps) - 1:
                cx = w // 2
                draw.line([(cx, y + row_h + 2), (cx, y + row_h + 16)], fill=ARROW_GRAY, width=2)
                draw.polygon(
                    [(cx - 4, y + row_h + 14), (cx + 4, y + row_h + 14), (cx, y + row_h + 19)],
                    fill=ARROW_GRAY,
                )
            y += row_h + 20
        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _logic_pipeline_horizontal(self, steps: list[dict[str, str]], title: str, name: str) -> Path:
        """横向方法流程：适合 GRASP 等多模块 pipeline."""
        n = len(steps)
        font_title = _load_font(18, bold=True)
        font_name = _load_font(12, bold=True)
        font_detail = _load_font(10)
        w = max(1100, 170 * n + 48)
        box_w = max(155, (w - 48 - (n - 1) * 28) // max(n, 1))
        h = 260
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        tw = _text_w(title, font_title)
        draw.text(((w - tw) // 2, 10), title, fill=(35, 45, 60), font=font_title)
        gap = 28
        start_x = (w - (n * box_w + (n - 1) * gap)) // 2
        y = 42
        for i, step in enumerate(steps):
            x = start_x + i * (box_w + gap)
            color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            name_lines = _wrap_to_width(step["name"], font_name, box_w - 16)[:2]
            detail_lines = _wrap_to_width(step.get("detail", ""), font_detail, box_w - 16)[:3]
            box_h = max(120, 24 + len(name_lines) * 14 + len(detail_lines) * 11)
            _draw_shadow_box(draw, x, y, box_w, box_h, color, radius=FLOW_RADIUS)
            draw.rectangle([x, y, x + 6, y + box_h], fill=color)
            cy = y + 8
            for line in name_lines:
                draw.text((x + 12, cy), line, fill=(25, 35, 50), font=font_name)
                cy += 14
            for line in detail_lines:
                draw.text((x + 12, cy), line, fill=(60, 68, 82), font=font_detail)
                cy += 11
            if i < n - 1:
                ax = x + box_w + 4
                mid_y = y + box_h // 2
                draw.line([(ax, mid_y), (ax + gap - 8, mid_y)], fill=(120, 130, 145), width=2)
                draw.polygon(
                    [(ax + gap - 12, mid_y - 5), (ax + gap - 12, mid_y + 5), (ax + gap - 4, mid_y)],
                    fill=(120, 130, 145),
                )
        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

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
        layout = str(spec.data.get("layout", "")).lower()
        layers = spec.data.get("layers") or []
        if not layers:
            steps = [format_body(str(s)) for s in (spec.data.get("steps") or [])]
            layers = _steps_to_layers(steps)
        if not layers or _layers_too_generic(layers):
            layers = _steps_to_layers(
                [format_body(str(s)) for s in (spec.data.get("steps") or ["Input", "Process", "Output"])]
            )

        if layout == "horizontal" and len(layers) >= 4:
            return self._architecture_horizontal(layers, smart_title(spec.title or "Architecture"), name)

        font_title = _load_font(16, bold=True)
        font_name = _load_font(13, bold=True)
        font_detail = _load_font(11)
        w = 640
        box_w = w - 28
        inner_w = box_w - 20
        y = 34
        row_heights: list[int] = []
        for layer in layers:
            name_text = str(layer.get("name", "Block"))
            detail = str(layer.get("detail", ""))
            name_lines = _wrap_to_width(name_text, font_name, inner_w)[:1]
            detail_lines = _wrap_to_width(detail, font_detail, inner_w)[:3]
            row_h = 14 + len(name_lines) * 14 + len(detail_lines) * 12 + 10
            row_heights.append(max(row_h, 44))
            y += row_h + 10
        h = y + 8

        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        tw = _text_w(smart_title(spec.title or "Architecture"), font_title)
        draw.text(((w - tw) // 2, 8), smart_title(spec.title or "Architecture"), fill=(35, 45, 60), font=font_title)

        y = 34
        for i, layer in enumerate(layers):
            color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            row_h = row_heights[i]
            _draw_shadow_box(draw, 14, y, box_w, row_h, color, radius=FLOW_RADIUS)
            draw.rectangle([14, y, 14 + 6, y + row_h], fill=color)
            cy = y + 8
            for line in _wrap_to_width(str(layer.get("name", "")), font_name, inner_w)[:1]:
                draw.text((26, cy), line, fill=(30, 40, 55), font=font_name)
                cy += 14
            for line in _wrap_to_width(str(layer.get("detail", "")), font_detail, inner_w)[:3]:
                draw.text((26, cy), line, fill=(70, 78, 92), font=font_detail)
                cy += 12
            if i < len(layers) - 1:
                cx = w // 2
                draw.line([(cx, y + row_h + 1), (cx, y + row_h + 9)], fill=(130, 140, 155), width=2)
            y += row_h + 10

        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _architecture_horizontal(self, layers: list[dict], title: str, name: str) -> Path:
        n = len(layers)
        font_title = _load_font(18, bold=True)
        font_name = _load_font(12, bold=True)
        font_detail = _load_font(10)
        w = max(1050, 165 * n + 40)
        box_w = max(150, (w - 40 - (n - 1) * 24) // max(n, 1))
        h = 240
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)
        tw = _text_w(title, font_title)
        draw.text(((w - tw) // 2, 8), title, fill=(35, 45, 60), font=font_title)
        gap = 24
        start_x = (w - (n * box_w + (n - 1) * gap)) // 2
        y = 38
        for i, layer in enumerate(layers):
            x = start_x + i * (box_w + gap)
            color = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            name_lines = _wrap_to_width(str(layer.get("name", "")), font_name, box_w - 14)[:2]
            detail_lines = _wrap_to_width(str(layer.get("detail", "")), font_detail, box_w - 14)[:2]
            box_h = max(110, 22 + len(name_lines) * 14 + len(detail_lines) * 11)
            _draw_shadow_box(draw, x, y, box_w, box_h, color, radius=FLOW_RADIUS)
            draw.rectangle([x, y, x + 5, y + box_h], fill=color)
            cy = y + 8
            for line in name_lines:
                draw.text((x + 10, cy), line, fill=(30, 40, 55), font=font_name)
                cy += 14
            for line in detail_lines:
                draw.text((x + 10, cy), line, fill=(70, 78, 92), font=font_detail)
                cy += 11
            if i < n - 1:
                ax = x + box_w + 3
                mid_y = y + box_h // 2
                draw.line([(ax, mid_y), (ax + gap - 6, mid_y)], fill=(130, 140, 155), width=2)
        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _bullet_cards(self, spec: VisualSpec, name: str) -> Path:
        """结论等区块：编号要点卡，文字换行不截断."""
        raw_items = spec.data.get("items") or spec.data.get("steps") or []
        items = [_normalize_card_item(x) for x in raw_items][:3]
        items = [format_body(x) for x in items if x.strip()]
        if not items:
            items = ["Takeaway"]

        w = 540
        n = len(items)
        pad = 12
        gap = 10
        card_w = (w - pad * 2 - gap * (n - 1)) // n
        card_w = max(card_w, 100)
        font_idx = _load_font(18, bold=True)
        font_txt = _load_font(10)
        inner = card_w - 16
        text_w = inner - 28

        row_heights: list[int] = []
        wrapped: list[list[str]] = []
        for item in items:
            lines = _wrap_to_width(item, font_txt, text_w)[:5]
            wrapped.append(lines)
            row_heights.append(max(72, 36 + len(lines) * 13))
        card_h = max(row_heights)
        h = card_h + 28

        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)

        for i, lines in enumerate(wrapped):
            accent = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            x = pad + i * (card_w + gap)
            y = 12
            _draw_shadow_box(draw, x, y, card_w, card_h, accent, radius=CARD_RADIUS, fill=(255, 255, 255))
            draw.ellipse([x + 8, y + 8, x + 28, y + 28], fill=accent)
            idx = str(i + 1)
            iw = _text_w(idx, font_idx)
            draw.text((x + 18 - iw // 2, y + 10), idx, fill=(255, 255, 255), font=font_idx)
            ty = y + 34
            for line in lines:
                draw.text((x + 8, ty), line, fill=(45, 50, 60), font=font_txt)
                ty += 13

        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _data_table(self, spec: VisualSpec, name: str) -> Path:
        """渲染论文关键实验数据表."""
        headers = [str(h) for h in spec.data.get("headers", [])][:5]
        rows = [[str(c) for c in row] for row in spec.data.get("rows", [])][:6]
        caption = smart_title(str(spec.data.get("caption") or spec.title or "Results Table"))
        if not headers or not rows:
            headers = ["Method", "Metric"]
            rows = [["—", "—"]]

        n_cols = len(headers)
        w = 520
        pad = 10
        col_w = (w - pad * 2) // n_cols
        col_w = max(col_w, 72)
        w = col_w * n_cols + pad * 2

        font_cap = _load_font(11, bold=True)
        font_hdr = _load_font(10, bold=True)
        font_cell = _load_font(9)
        row_h = 22
        cap_lines = _wrap_to_width(caption, font_cap, w - pad * 2)[:2]
        h = pad + len(cap_lines) * 14 + 8 + row_h * (len(rows) + 1) + pad

        img = Image.new("RGB", (w, h), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        y = pad
        for line in cap_lines:
            draw.text((pad, y), line, fill=FLOW_TITLE, font=font_cap)
            y += 14
        y += 4
        hdr_y = y
        for j, hdr in enumerate(headers):
            x = pad + j * col_w
            draw.rectangle([x, hdr_y, x + col_w, hdr_y + row_h], fill=ACCENT_RGB)
            txt = truncate_cell(hdr, font_hdr, col_w - 8)
            draw.text((x + 6, hdr_y + 5), txt, fill=(255, 255, 255), font=font_hdr)
        y = hdr_y + row_h
        for i, row in enumerate(rows):
            fill = (255, 255, 255) if i % 2 == 0 else BG_TABLE_ALT
            for j in range(n_cols):
                x = pad + j * col_w
                draw.rectangle([x, y, x + col_w, y + row_h], fill=fill)
                if j == 0:
                    draw.line([(x, y), (x, y + row_h)], fill=(220, 220, 220), width=1)
                draw.line([(x, y + row_h - 1), (x + col_w, y + row_h - 1)], fill=(220, 220, 220), width=1)
                val = row[j] if j < len(row) else ""
                draw.text((x + 6, y + 5), truncate_cell(val, font_cell, col_w - 10), fill=(40, 40, 40), font=font_cell)
            y += row_h

        out = self.output_dir / f"{name}.png"
        img.save(out)
        return out

    def _stat_cards(self, spec: VisualSpec, name: str) -> Path:
        cards = _normalize_stat_cards(spec.data.get("cards", [
            {"label": "Metric A", "value": "92", "unit": "%"},
            {"label": "Metric B", "value": "4.5", "unit": "/5"},
        ]))[:3]

        w = 540
        n = max(len(cards), 1)
        pad = 14
        gap = 12
        card_w = (w - pad * 2 - gap * (n - 1)) // n
        card_w = max(card_w, 100)

        val_size = 28 if card_w >= 130 else 24 if card_w >= 110 else 20
        lbl_size = 11 if card_w >= 110 else 10
        font_val = _load_font(val_size, bold=True)
        font_lbl = _load_font(lbl_size)
        inner_pad = 10
        text_w_max = card_w - inner_pad * 2

        layouts: list[tuple[list[str], list[str], int]] = []
        max_card_h = 120
        for card in cards:
            val = str(card.get("value", ""))
            unit = str(card.get("unit", ""))
            val_lines = _wrap_to_width(f"{val}{unit}".strip(), font_val, text_w_max)[:2]
            label_lines = _wrap_to_width(str(card.get("label", "")), font_lbl, text_w_max)[:3]
            val_block = len(val_lines) * (val_size + 4)
            lbl_block = len(label_lines) * (lbl_size + 4)
            card_h = inner_pad * 2 + val_block + 8 + lbl_block
            max_card_h = max(max_card_h, card_h)
            layouts.append((val_lines, label_lines, card_h))

        h = max_card_h + 32
        img = Image.new("RGB", (w, h), (248, 250, 252))
        draw = ImageDraw.Draw(img)

        for i, (card, (val_lines, label_lines, _)) in enumerate(zip(cards, layouts)):
            accent = _hex_to_rgb(PALETTE[i % len(PALETTE)])
            x = pad + i * (card_w + gap)
            y = 16
            _draw_shadow_box(draw, x, y, card_w, max_card_h, accent, radius=CARD_RADIUS, fill=(255, 255, 255))
            draw.rectangle([x, y, x + card_w, y + 6], fill=accent)

            vy = y + inner_pad + 4
            for line in val_lines:
                draw.text((x + inner_pad, vy), line, fill=accent, font=font_val)
                vy += val_size + 4

            line_h = lbl_size + 4
            block_h = len(label_lines) * line_h
            start_y = y + max_card_h - inner_pad - block_h
            for j, line in enumerate(label_lines):
                draw.text((x + inner_pad, start_y + j * line_h), line, fill=(60, 65, 75), font=font_lbl)

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


def _draw_academic_box(draw: ImageDraw.ImageDraw, x: int, y: int, w: int, h: int) -> None:
    """白底黑框步骤盒（参考海报 Methods 流程图）."""
    draw.rectangle([x, y, x + w, y + h], fill=FLOW_FILL, outline=FLOW_BORDER, width=2)


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


def truncate_cell(text: str, font, max_w: int) -> str:
    text = str(text).strip()
    if _text_w(text, font) <= max_w:
        return text
    while text and _text_w(text + "…", font) > max_w:
        text = text[:-1]
    return (text + "…") if text else "—"


def _normalize_card_item(item) -> str:
    if isinstance(item, dict):
        parts: list[str] = []
        for key in ("content", "title", "text", "body", "bullet", "summary", "description"):
            val = item.get(key)
            if val and str(val).strip():
                parts.append(str(val).strip())
        if parts:
            return " — ".join(parts)
        for val in item.values():
            if isinstance(val, str) and len(val.strip()) > 12:
                return val.strip()
        return ""
    text = str(item).strip()
    if text.startswith("{") and "content" in text:
        import ast
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, dict):
                return _normalize_card_item(parsed)
        except (ValueError, SyntaxError):
            pass
    return text


def _normalize_stat_cards(cards: list) -> list[dict]:
    out: list[dict] = []
    for i, card in enumerate(cards):
        if isinstance(card, dict):
            out.append(card)
        elif isinstance(card, str):
            out.append({"label": f"Item {i + 1}", "value": card, "unit": ""})
    return out or [{"label": "Metric A", "value": "92", "unit": "%"}]


def _steps_to_layers(steps: list[str]) -> list[dict[str, str]]:
    default_names = ["Encoder", "Decoder", "Attention", "Feed-Forward"]
    layers: list[dict[str, str]] = []
    for i, step in enumerate(steps[:4]):
        text = step.strip()
        if ":" in text:
            name, detail = text.split(":", 1)
        elif "—" in text:
            name, detail = text.split("—", 1)
        elif "-" in text and len(text.split("-", 1)[0]) < 20:
            name, detail = text.split("-", 1)
        else:
            name = default_names[i] if i < len(default_names) else f"Block {i + 1}"
            detail = text
        layers.append({"name": name.strip()[:28], "detail": detail.strip()[:90]})
    return layers


def _layers_too_generic(layers: list[dict[str, str]]) -> bool:
    generic = {"input", "process", "output", "step 1", "step 2", "step 3"}
    names = {str(l.get("name", "")).strip().lower() for l in layers}
    return names.issubset(generic) or len(names) <= 1


def _short_label(text: str, max_len: int) -> str:
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def _truncate_to_width(text: str, font, max_w: int) -> str:
    if _text_w(text, font) <= max_w:
        return text
    ell = "…"
    while text and _text_w(text + ell, font) > max_w:
        text = text[:-1]
    return (text + ell) if text else ell


def _wrap_to_width(text: str, font, max_w: int) -> list[str]:
    if max_w < 24:
        return [_truncate_to_width(text, font, max_w)]
    words = text.split()
    if not words:
        return [""]
    lines: list[str] = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if _text_w(trial, font) <= max_w:
            cur = trial
        else:
            lines.append(_truncate_to_width(cur, font, max_w))
            cur = word
    lines.append(_truncate_to_width(cur, font, max_w))
    return lines


def _format_bar_value(val) -> str:
    try:
        v = float(val)
    except (TypeError, ValueError):
        return str(val)
    if abs(v) < 10 and abs(v - round(v)) > 0.001:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.0f}" if isinstance(val, float) else str(val)


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
