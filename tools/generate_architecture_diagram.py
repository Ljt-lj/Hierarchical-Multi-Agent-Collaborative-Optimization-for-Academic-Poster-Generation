"""Generate PosterAgent architecture flowchart (ICLR-style) for PPT / paper."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path as MPath

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures" / "posteragent_architecture.png"

C = {
    "bg": "#FAFBFE",
    "phase1": "#FFF7ED",
    "phase1_edge": "#FDBA74",
    "phase2": "#F0F4FF",
    "phase2_edge": "#93B4F4",
    "agent": "#EEF4FF",
    "agent_edge": "#4F7FD0",
    "data": "#ECFDF3",
    "data_edge": "#34A06E",
    "stage": "#FFF1E6",
    "stage_edge": "#E07A2F",
    "ctrl": "#FEF3C7",
    "ctrl_edge": "#D97706",
    "accent": "#6366F1",
    "text": "#1E293B",
    "muted": "#64748B",
    "white": "#FFFFFF",
    "arrow": "#64748B",
}


def _patch(ax, patch, zorder=3):
    patch.set_transform(ax.transAxes)
    ax.add_patch(patch)


def icon_pdf(ax, x, y, s=0.014, color="#E07A2F"):
    w, h = s * 0.75, s
    _patch(ax, FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.002", fc=color, ec="white", lw=0.8), 6)
    for yy in [y - h * 0.15, y, y + h * 0.15]:
        _patch(ax, Rectangle((x - w * 0.28, yy - 0.0012), w * 0.56, 0.0025, fc="white", alpha=0.85), 6)


def icon_tree(ax, x, y, s=0.014, color="#34A06E"):
    r = s * 0.14
    nodes = [(x, y + s * 0.28), (x - s * 0.28, y - s * 0.08), (x + s * 0.28, y - s * 0.08)]
    for nx, ny in [(x, y + s * 0.08), (x - s * 0.14, y - s * 0.2), (x + s * 0.14, y - s * 0.2)]:
        _patch(ax, FancyArrowPatch((nodes[0][0], nodes[0][1] - r), (nx, ny + r), arrowstyle="-", color=color, lw=1.0, mutation_scale=0), 5)
    for nx, ny in nodes:
        _patch(ax, Circle((nx, ny), r, fc=color, ec="white", lw=0.6), 6)


def icon_llm(ax, x, y, s=0.014, color="#4F7FD0"):
    r = s * 0.1
    layers = [[(x, y + s * 0.22)], [(x - s * 0.16, y), (x + s * 0.16, y)], [(x, y - s * 0.22)]]
    prev = []
    for layer in layers:
        for xx, yy in layer:
            _patch(ax, Circle((xx, yy), r, fc=color, ec="white", lw=0.6), 6)
            for px, py in prev:
                _patch(ax, FancyArrowPatch((px, py - r), (xx, yy + r), arrowstyle="-", color=color, lw=0.7, alpha=0.45, mutation_scale=0), 5)
        prev = layer


def icon_image(ax, x, y, s=0.014, color="#4F7FD0"):
    w, h = s * 0.9, s * 0.65
    _patch(ax, FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.002", fc="white", ec=color, lw=1.0), 6)
    _patch(ax, Circle((x + w * 0.18, y + h * 0.12), s * 0.06, fc=color, alpha=0.5), 6)


def icon_columns(ax, x, y, s=0.014, color="#4F7FD0"):
    w, h, g = s * 0.15, s * 0.45, s * 0.06
    for i in range(3):
        _patch(ax, FancyBboxPatch((x - w - g + i * (w + g), y - h / 2), w, h, boxstyle="round,pad=0.001", fc=color, ec="white", lw=0.5, alpha=0.85), 6)


def icon_brush(ax, x, y, s=0.014, color="#4F7FD0"):
    _patch(ax, FancyBboxPatch((x - s * 0.1, y - s * 0.28), s * 0.2, s * 0.35, boxstyle="round,pad=0.002", fc=color, ec="white", lw=0.6), 6)


def icon_check(ax, x, y, s=0.014, color="#34A06E"):
    _patch(ax, Circle((x, y), s * 0.32, fc=color, ec="white", lw=0.7), 6)


def icon_grid(ax, x, y, s=0.014, color="#4F7FD0"):
    w = s * 0.18
    for i in range(2):
        for j in range(2):
            _patch(ax, Rectangle((x - s * 0.22 + j * (w + 0.003), y - s * 0.22 + i * (w + 0.003)), w, w, fc=color, ec="white", lw=0.4, alpha=0.75), 6)


def icon_poster(ax, x, y, s=0.014, color="#4F7FD0"):
    w, h = s * 0.5, s * 0.65
    _patch(ax, FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.002", fc="white", ec=color, lw=1.0), 6)


def icon_warning(ax, x, y, s=0.014, color="#D97706"):
    tri = [(x, y + s * 0.3), (x - s * 0.28, y - s * 0.22), (x + s * 0.28, y - s * 0.22)]
    _patch(ax, PathPatch(MPath(tri + [tri[0]], [MPath.MOVETO, MPath.LINETO, MPath.LINETO, MPath.CLOSEPOLY]), fc=color, ec="white", lw=0.7), 6)


def icon_chat(ax, x, y, s=0.014, color="#4F7FD0"):
    _patch(ax, FancyBboxPatch((x - s * 0.28, y - s * 0.06), s * 0.56, s * 0.28, boxstyle="round,pad=0.003", fc=color, ec="white", lw=0.6), 6)


def icon_dial(ax, x, y, s=0.014, color="#D97706"):
    _patch(ax, Circle((x, y), s * 0.3, fc=color, ec="white", lw=0.8), 6)


def icon_output(ax, x, y, s=0.014, color="#E07A2F"):
    w, h = s * 0.55, s * 0.72
    _patch(ax, FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.003", fc=color, ec="white", lw=0.9), 6)


ICON_MAP = {
    "pdf": icon_pdf, "parser": icon_llm, "tree": icon_tree, "planner": icon_llm, "plan": icon_tree,
    "refiner": icon_llm, "expander": icon_llm, "visual": icon_image, "balancer": icon_columns,
    "painter": icon_brush, "semantic": icon_check, "layout": icon_grid, "renderer": icon_poster,
    "error": icon_warning, "commenter": icon_chat, "controller": icon_dial, "output": icon_output,
}


def node_box(ax, xy, w, h, title, subtitle=None, kind="agent", icon_key=None, badge=None):
    x, y = xy
    fc, ec = C[kind], C[f"{kind}_edge"]
    _patch(ax, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.014",
                              linewidth=1.4, edgecolor=ec, facecolor=fc, zorder=5))
    if badge:
        _patch(ax, Circle((x + 0.011, y + h - 0.011), 0.010, fc=C["accent"], ec="white", lw=0.7, zorder=8))
        ax.text(x + 0.011, y + h - 0.011, badge, ha="center", va="center", fontsize=7,
                fontweight="bold", color="white", transform=ax.transAxes, zorder=9)
    cx = x + w / 2
    if icon_key and icon_key in ICON_MAP:
        ICON_MAP[icon_key](ax, cx, y + h * 0.72, s=min(w, h) * 0.30, color=ec)
    if subtitle:
        ax.text(cx, y + h * 0.46, title, ha="center", va="center", fontsize=9, fontweight="bold",
                color=C["text"], transform=ax.transAxes, zorder=7)
        ax.text(cx, y + h * 0.18, subtitle, ha="center", va="center", fontsize=7.5,
                color=C["muted"], transform=ax.transAxes, zorder=7)
    else:
        ax.text(cx, y + h * 0.40, title, ha="center", va="center", fontsize=9.2, fontweight="bold",
                color=C["text"], transform=ax.transAxes, zorder=7)
    return x + w / 2, y + h / 2, w, h


def seg(ax, p0, p1, color=C["arrow"], lw=1.4, dashed=False, arrow_end=False, zorder=2):
    style = "-|>" if arrow_end else "-"
    arr = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=11, linewidth=lw, color=color,
        transform=ax.transAxes, zorder=zorder,
        linestyle=(0, (4, 3)) if dashed else "solid",
        connectionstyle="arc3,rad=0",
    )
    ax.add_patch(arr)


def route(ax, points, color=C["arrow"], lw=1.4, dashed=False, arrow_end=True, zorder=2):
    for i in range(len(points) - 1):
        seg(ax, points[i], points[i + 1], color=color, lw=lw, dashed=dashed,
            arrow_end=(arrow_end and i == len(points) - 2), zorder=zorder)


def label(ax, x, y, text, color=C["muted"], fontsize=8, ha="center", va="center"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, transform=ax.transAxes, zorder=10,
            bbox=dict(boxstyle="round,pad=0.28", fc=C["white"], ec="#E2E8F0", alpha=0.98, lw=0.5))


def phase_block(ax, xy, w, h, fc, ec, title, subtitle=None):
    """Phase container with internal header strip — no title/border overlap."""
    x, y = xy
    _patch(ax, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.018",
                              linewidth=1.4, edgecolor=ec, facecolor=fc, alpha=0.45, zorder=1))
    if not title and not subtitle:
        return y + 0.012, h - 0.024
    header_h = 0.062
    header_y = y + h - header_h
    _patch(ax, FancyBboxPatch((x + 0.005, header_y), w - 0.010, header_h - 0.004,
                              boxstyle="round,pad=0.004,rounding_size=0.010",
                              linewidth=0, facecolor=C["bg"], alpha=0.94, zorder=2))
    ax.text(x + 0.016, y + h - 0.008, title, ha="left", va="top", fontsize=11.2,
            fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=12)
    if subtitle:
        ax.text(x + 0.016, header_y + 0.010, subtitle, ha="left", va="bottom", fontsize=7.8,
                color=C["muted"], transform=ax.transAxes, zorder=12)
    content_y = y + 0.008
    content_h = header_y - content_y - 0.008
    return content_y, content_h


def legend_box(ax, x, y, w, h):
    """Vertical legend card — equal-row layout, chip + label aligned."""
    _patch(ax, FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.010,rounding_size=0.012",
                              fc=C["white"], ec="#E2E8F0", lw=1.1, alpha=0.98, zorder=4))
    ax.text(x + w / 2, y + h * 0.935, "图例", ha="center", va="center", fontsize=9.5,
            fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=12)

    items = [("输入/输出", "stage"), ("Agent", "agent"), ("中间结构", "data"), ("控制器", "ctrl")]
    chip_w, chip_h = 0.032, 0.024
    chip_x = x + w * 0.10
    text_x = x + w * 0.10 + chip_w + w * 0.05
    row_centers = [y + h * f for f in (0.78, 0.60, 0.42, 0.24)]

    for (lab, kind), cy in zip(items, row_centers):
        _patch(ax, FancyBboxPatch((chip_x, cy - chip_h / 2), chip_w, chip_h,
                                  boxstyle="round,pad=0.003", fc=C[kind], ec=C[f"{kind}_edge"],
                                  lw=1.3, zorder=11))
        ax.text(text_x, cy, lab, ha="left", va="center", fontsize=8.6,
                color=C["text"], transform=ax.transAxes, zorder=12)

    ax.text(x + w / 2, y + h * 0.065, "PosterAgent Architecture", ha="center", va="center",
            fontsize=7.2, color="#94A3B8", transform=ax.transAxes, zorder=12)


def row_tag(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="bottom", fontsize=8, fontweight="bold", color=C["muted"],
            transform=ax.transAxes, zorder=12,
            bbox=dict(boxstyle="round,pad=0.22", fc=C["bg"], ec="#E2E8F0", alpha=0.98, lw=0.5))


def main() -> None:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])

    # ---- title ----
    ax.text(0.5, 0.990, "PosterAgent：层次化多智能体学术海报生成架构", ha="center", va="top",
            fontsize=15.5, fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=12)
    ax.text(0.5, 0.958, "Training-free Hierarchical Multi-Agent Collaborative Optimization",
            ha="center", va="top", fontsize=9.5, color=C["muted"], transform=ax.transAxes, zorder=12)

    # layout — wider main column + full-height right panel
    phase_x, phase_w = 0.042, 0.668
    left = 0.060
    bw, bh, gap = 0.094, 0.084, 0.011
    bus_x = 0.016
    right_bus = 0.678

    # ---- Phase 1 (lower — clear gap below main subtitle) ----
    p1_y0 = 0.748
    p1_base, p1_inner = phase_block(
        ax, (phase_x, p1_y0), phase_w, 0.162, C["phase1"], C["phase1_edge"],
        "Phase 1 · 解析与逻辑规划", "PDF → Raw Tree → LogicPlan（logic_links + 语义约束）",
    )
    p1_y = p1_base + (p1_inner - bh) / 2
    p1_xs = [left + i * (bw + gap) for i in range(5)]
    p1_spec = [
        ("PDF 输入", None, "stage", "pdf", "1"),
        ("Parser", "Agent", "agent", "parser", "2"),
        ("Raw Tree", "Catalog", "data", "tree", "3"),
        ("LogicPlanner", "Agent", "agent", "planner", "4"),
        ("LogicPlan", "logic_links", "data", "plan", "5"),
    ]
    p1c = []
    for x, (t, s, k, ic, b) in zip(p1_xs, p1_spec):
        cx, cy, _, _ = node_box(ax, (x, p1_y), bw, bh, t, s, kind=k, icon_key=ic, badge=b)
        p1c.append((cx, cy, x))
    for i in range(4):
        seg(ax, (p1c[i][0] + bw / 2 - 0.003, p1c[i][1]), (p1c[i + 1][0] - bw / 2 + 0.003, p1c[i + 1][1]),
            arrow_end=True, zorder=2)

    # ---- Phase 2 (taller — fills lower canvas) ----
    p2_y0 = 0.248
    p2_base, p2_inner = phase_block(
        ax, (phase_x, p2_y0), phase_w, 0.458, C["phase2"], C["phase2_edge"],
        "Phase 2 · 自适应迭代优化环", "τ=0.85 early stop  ·  max 5 iter  ·  keep best score",
    )
    row2_y = p2_base + 0.024
    row1_y = p2_base + p2_inner - bh - 0.024
    row1_spec = [
        ("Refiner", "Agent", "refiner", "6"),
        ("Section Exp.", "Agent", "expander", "7"),
        ("Visual", "figure crop", "visual", "8"),
        ("Col. Balancer", "25/50/25", "balancer", "9"),
        ("Painter", "style", "painter", "10"),
    ]
    row2_spec = [
        ("Semantic", "Check", "semantic", "11"),
        ("Layout", "3-column", "layout", "12"),
        ("Renderer", "PNG/PPTX", "renderer", "13"),
        ("Error Anal.", "routing", "error", "14"),
        ("Commenter", "Agent", "commenter", "15"),
        ("Controller", "early stop", "controller", "16"),
    ]

    row_tag(ax, left, row1_y + bh + 0.008, "Row A · 内容精炼 & 视觉")
    row_tag(ax, left, row2_y + bh + 0.008, "Row B · 布局 & 评估")

    r1_xs = [left + i * (bw + gap) for i in range(5)]
    r1c = []
    for x, (t, s, ic, b) in zip(r1_xs, row1_spec):
        cx, cy, _, _ = node_box(ax, (x, row1_y), bw, bh, t, s, icon_key=ic, badge=b)
        r1c.append((cx, cy, x))

    r2_xs = [left + i * (bw + gap) for i in range(6)]
    r2c = []
    for x, (t, s, ic, b) in zip(r2_xs, row2_spec):
        kind = "ctrl" if ic == "controller" else "agent"
        cx, cy, _, _ = node_box(ax, (x, row2_y), bw, bh, t, s, kind=kind, icon_key=ic, badge=b)
        r2c.append((cx, cy, x))

    for i in range(4):
        seg(ax, (r1c[i][0] + bw / 2 - 0.003, r1c[i][1]), (r1c[i + 1][0] - bw / 2 + 0.003, r1c[i + 1][1]),
            arrow_end=True, zorder=2)
    for i in range(5):
        seg(ax, (r2c[i][0] + bw / 2 - 0.003, r2c[i][1]), (r2c[i + 1][0] - bw / 2 + 0.003, r2c[i + 1][1]),
            arrow_end=True, zorder=2)

    # Painter -> Semantic via right bus (avoid crossing boxes)
    py_top = row1_y + bh
    sy_bot = row2_y
    route(ax, [
        (r1c[-1][0], py_top + 0.004),
        (right_bus, py_top + 0.004),
        (right_bus, sy_bot + bh - 0.004),
        (r2c[0][0], sy_bot + bh - 0.004),
    ], zorder=2)

    # ---- init: LogicPlan -> Refiner via left bus (no diagonal) ----
    lp_x, lp_y = p1c[-1][0], p1_y
    rf_x, rf_y = r1c[0][0], row1_y + bh
    bridge_y = (p1_y0 + p2_y0 + 0.458) / 2
    route(ax, [
        (lp_x, lp_y + 0.004),
        (lp_x, bridge_y),
        (bus_x, bridge_y),
        (bus_x, rf_y + 0.004),
        (rf_x - bw / 2 + 0.004, rf_y + 0.004),
    ], color=C["phase1_edge"], lw=1.5, zorder=2)
    label(ax, 0.12, bridge_y + 0.010, "初始化", ha="center", fontsize=8, color=C["phase1_edge"])

    # ---- feedback loop via bottom corridor ----
    loop_y = p2_base + 0.012
    ctrl_x = r2c[-1][0]
    ref_x = r1c[0][0]
    route(ax, [
        (ctrl_x, row2_y + 0.004),
        (ctrl_x, loop_y),
        (ref_x, loop_y),
        (ref_x, row1_y + 0.004),
    ], color=C["ctrl_edge"], lw=1.5, dashed=True, zorder=2)
    label(ax, (ctrl_x + ref_x) / 2, loop_y - 0.018, "未达标 → 下一轮", color=C["ctrl_edge"])

    # ---- output (closer to phase2, fills bottom) ----
    out_w, out_h = 0.150, 0.086
    out_x, out_y = 0.275, 0.118
    node_box(ax, (out_x, out_y), out_w, out_h, "Best Poster", "PNG + PPTX", kind="stage", icon_key="output")
    out_cx = out_x + out_w / 2
    route(ax, [
        (ctrl_x, row2_y + 0.004),
        (ctrl_x, out_y + out_h + 0.012),
        (out_cx, out_y + out_h + 0.012),
        (out_cx, out_y + out_h - 0.004),
    ], color=C["stage_edge"], lw=1.5, zorder=2)
    label(ax, out_cx + 0.11, out_y + out_h + 0.022, "score ≥ τ\n或达最大轮次", color=C["stage_edge"], fontsize=7.8)

    # ---- right info panel (full height: Phase1 bottom → Phase2 bottom) ----
    info_x, info_w = 0.728, 0.248
    info_y = p2_y0
    info_h = (p1_y0 + 0.162) - p2_y0
    _patch(ax, FancyBboxPatch((info_x, info_y), info_w, info_h, boxstyle="round,pad=0.012,rounding_size=0.015",
                              fc=C["white"], ec="#E2E8F0", lw=1.2, alpha=0.98, zorder=4))
    ax.text(info_x + info_w / 2, info_y + info_h - 0.022, "评分 · 三树 · 错误路由",
            ha="center", va="top", fontsize=10.5, fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=10)
    ax.text(info_x + 0.014, info_y + info_h - 0.052,
            "overall = 0.4·Sem + 0.35·LB + 0.25·ITM",
            ha="left", va="top", fontsize=9, color=C["accent"], fontweight="bold",
            transform=ax.transAxes, zorder=10)
    info_lines = [
        ("三树表示", True),
        ("Raw Tree      ← Parser", False),
        ("Content Tree  ← Refiner", False),
        ("Poster Tree   ← Layout", False),
        ("错误路由", True),
        ("overflow  →  Layout", False),
        ("缺图      →  Visual", False),
        ("逻辑      →  Refiner", False),
    ]
    # distribute lines evenly in panel body
    body_top = info_y + info_h - 0.088
    body_bot = info_y + 0.028
    n_lines = len(info_lines)
    step = (body_top - body_bot) / (n_lines - 1) if n_lines > 1 else 0
    for i, (line, is_head) in enumerate(info_lines):
        yy = body_top - i * step
        ax.text(info_x + 0.014, yy, line, ha="left", va="top",
                fontsize=9.2 if is_head else 8.6,
                fontweight="bold" if is_head else "normal",
                color=C["text"] if is_head else C["muted"],
                transform=ax.transAxes, zorder=10)

    # ---- legend (bottom-right card, aligned with info panel) ----
    legend_box(ax, info_x, 0.018, info_w, 0.128)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=220, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.08)
    plt.close(fig)
    print(f"Saved: {OUT}")


if __name__ == "__main__":
    main()
