"""Generate Slide 5 flowcharts: Phase 1 and Phase 2 (standalone for PPT)."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from generate_architecture_diagram import (
    C,
    ROOT,
    label,
    node_box,
    phase_block,
    route,
    row_tag,
    seg,
)

OUT_P1 = ROOT / "paper" / "figures" / "slide5_phase1_flow.png"
OUT_P2 = ROOT / "paper" / "figures" / "slide5_phase2_flow.png"


def _setup_ax(figsize):
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["bg"])
    return fig, ax


def draw_phase1_flow(out: Path = OUT_P1) -> Path:
    """Phase 1: PDF → Parser → Raw Tree → LogicPlanner → LogicPlan"""
    fig, ax = _setup_ax((14, 2.8))
    ax.text(0.5, 0.94, "Phase 1 · 解析与逻辑规划", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=12)
    ax.text(0.5, 0.78, "PDF → Raw Tree → LogicPlan（logic_links + 语义约束）",
            ha="center", va="top", fontsize=9.5, color=C["muted"], transform=ax.transAxes, zorder=12)

    frame_x, frame_w = 0.04, 0.82
    base, inner = phase_block(
        ax, (frame_x, 0.22), frame_w, 0.52, C["phase1"], C["phase1_edge"],
        "", None,
    )
    # hide duplicate inner header strip text by using empty title — frame still shows color
    bw, bh, gap = 0.138, 0.38, 0.018
    total_w = 5 * bw + 4 * gap
    left = frame_x + (frame_w - total_w) / 2
    ny = base + (inner - bh) / 2

    spec = [
        ("PDF 输入", None, "stage", "pdf", "1"),
        ("Parser", "Agent", "agent", "parser", "2"),
        ("Raw Tree", "Catalog", "data", "tree", "3"),
        ("LogicPlanner", "Agent", "agent", "planner", "4"),
        ("LogicPlan", "logic_links", "data", "plan", "5"),
    ]
    centers = []
    for i, (t, s, k, ic, b) in enumerate(spec):
        x = left + i * (bw + gap)
        cx, cy, _, _ = node_box(ax, (x, ny), bw, bh, t, s, kind=k, icon_key=ic, badge=b)
        centers.append((cx, cy))

    for i in range(4):
        seg(ax, (centers[i][0] + bw / 2 - 0.004, centers[i][1]),
            (centers[i + 1][0] - bw / 2 + 0.004, centers[i + 1][1]), arrow_end=True)

    # → Phase 2
    seg(ax, (centers[-1][0] + bw / 2 + 0.008, centers[-1][1]),
        (0.92, centers[-1][1]), color=C["phase1_edge"], lw=1.8, arrow_end=True)
    ax.text(0.935, centers[-1][1] + 0.12, "→ Phase 2", ha="left", va="center", fontsize=10,
            fontweight="bold", color=C["phase1_edge"], transform=ax.transAxes, zorder=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def draw_phase2_flow(out: Path = OUT_P2) -> Path:
    """Phase 2: adaptive iteration loop with feedback and output."""
    fig, ax = _setup_ax((14, 5.2))
    ax.text(0.5, 0.96, "Phase 2 · 自适应迭代优化环", ha="center", va="top",
            fontsize=14, fontweight="bold", color=C["text"], transform=ax.transAxes, zorder=12)
    ax.text(0.5, 0.905, "overall = 0.4·Sem + 0.35·LB + 0.25·ITM  ·  τ=0.85 early stop  ·  max 5 iter  ·  keep best",
            ha="center", va="top", fontsize=8.5, color=C["muted"], transform=ax.transAxes, zorder=12)

    frame_x, frame_w = 0.03, 0.94
    p2_y0 = 0.28
    base, inner = phase_block(
        ax, (frame_x, p2_y0), frame_w, 0.58, C["phase2"], C["phase2_edge"],
        "", None,
    )

    bw, bh, gap = 0.128, 0.22, 0.012
    row2_y = base + 0.03
    row1_y = base + inner - bh - 0.03
    left = frame_x + 0.04

    row_tag(ax, left, row1_y + bh + 0.012, "Row A · 内容精炼 & 视觉")
    row_tag(ax, left, row2_y + bh + 0.012, "Row B · 布局 & 评估")

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

    r1c, r2c = [], []
    for i, (t, s, ic, b) in enumerate(row1_spec):
        x = left + i * (bw + gap)
        cx, cy, _, _ = node_box(ax, (x, row1_y), bw, bh, t, s, icon_key=ic, badge=b)
        r1c.append((cx, cy))
    for i, (t, s, ic, b) in enumerate(row2_spec):
        x = left + i * (bw + gap)
        kind = "ctrl" if ic == "controller" else "agent"
        cx, cy, _, _ = node_box(ax, (x, row2_y), bw, bh, t, s, kind=kind, icon_key=ic, badge=b)
        r2c.append((cx, cy))

    for i in range(4):
        seg(ax, (r1c[i][0] + bw / 2 - 0.003, r1c[i][1]), (r1c[i + 1][0] - bw / 2 + 0.003, r1c[i + 1][1]),
            arrow_end=True)
    for i in range(5):
        seg(ax, (r2c[i][0] + bw / 2 - 0.003, r2c[i][1]), (r2c[i + 1][0] - bw / 2 + 0.003, r2c[i + 1][1]),
            arrow_end=True)

    right_bus = left + 5 * (bw + gap) - gap + 0.02
    route(ax, [
        (r1c[-1][0], row1_y + bh + 0.004),
        (right_bus, row1_y + bh + 0.004),
        (right_bus, row2_y + bh - 0.004),
        (r2c[0][0], row2_y + bh - 0.004),
    ])

    loop_y = base + 0.012
    ctrl_x, ref_x = r2c[-1][0], r1c[0][0]
    route(ax, [
        (ctrl_x, row2_y + 0.004),
        (ctrl_x, loop_y),
        (ref_x, loop_y),
        (ref_x, row1_y + 0.004),
    ], color=C["ctrl_edge"], lw=1.6, dashed=True)
    label(ax, (ctrl_x + ref_x) / 2, loop_y - 0.022, "未达标 → 下一轮", color=C["ctrl_edge"], fontsize=8)

    out_w, out_h = 0.16, 0.12
    out_x, out_y = 0.42, 0.06
    node_box(ax, (out_x, out_y), out_w, out_h, "Best Poster", "PNG + PPTX", kind="stage", icon_key="output")
    out_cx = out_x + out_w / 2
    route(ax, [
        (ctrl_x, row2_y + 0.004),
        (ctrl_x, out_y + out_h + 0.014),
        (out_cx, out_y + out_h + 0.014),
        (out_cx, out_y + out_h - 0.004),
    ], color=C["stage_edge"], lw=1.6)
    label(ax, out_cx + 0.14, out_y + out_h + 0.018, "score ≥ τ 或达最大轮次",
          color=C["stage_edge"], fontsize=7.5)

    # Phase 1 entry
    seg(ax, (0.02, row1_y + bh / 2), (left - 0.008, row1_y + bh / 2),
        color=C["phase1_edge"], lw=1.6, arrow_end=True)
    ax.text(0.025, row1_y + bh / 2 + 0.06, "← Phase 1", ha="left", va="center", fontsize=9,
            fontweight="bold", color=C["phase1_edge"], transform=ax.transAxes, zorder=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight", facecolor=C["bg"], pad_inches=0.06)
    plt.close(fig)
    print(f"Saved: {out}")
    return out


def main() -> None:
    draw_phase1_flow()
    draw_phase2_flow()


if __name__ == "__main__":
    main()
