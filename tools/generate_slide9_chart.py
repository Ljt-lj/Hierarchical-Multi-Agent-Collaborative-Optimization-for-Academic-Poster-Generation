"""Generate Slide 9 chart: Overall / ITM / logic vs iteration (GRASP v19)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import MultipleLocator

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"
OUT = OUT_DIR / "slide9_score_chart.png"
SCORES_DIR = ROOT / "outputs" / "Integrating_diverse_experimental_information_to_assist_prote"
VERSION = "v19"

C = {
    "bg": "#FAFBFE",
    "text": "#1E293B",
    "muted": "#64748B",
    "overall": "#6366F1",
    "itm": "#E07A2F",
    "logic": "#34A06E",
    "tau": "#D97706",
    "grid": "#E2E8F0",
    "table_head": "#EEF2FF",
    "table_best": "#FEF3C7",
}

FALLBACK = [
    {"iteration": 1, "overall": 0.742, "image_text_match": 0.85, "logic_score": 0.80},
    {"iteration": 2, "overall": 0.784, "image_text_match": 0.85, "logic_score": 1.00},
    {"iteration": 3, "overall": 0.808, "image_text_match": 0.90, "logic_score": 0.85},
]

TAU = 0.85


def load_history() -> list[dict]:
    if SCORES_DIR.exists():
        matches = sorted(SCORES_DIR.glob(f"*{VERSION}*_scores.json"))
        if matches:
            data = json.loads(matches[0].read_text(encoding="utf-8"))
            hist = data.get("history", [])
            if hist:
                return hist
    return FALLBACK


def _y_limits(overall: list[float], itm: list[float], logic: list[float]) -> tuple[float, float]:
    vals = overall + itm + logic + [TAU]
    lo, hi = min(vals), max(vals)
    pad = max((hi - lo) * 0.15, 0.035)
    y0 = math.floor((lo - pad) * 20) / 20
    y1 = math.ceil((hi + pad) * 20) / 20
    y0 = max(0.0, y0)
    y1 = min(1.05, max(y1, y0 + 0.22))
    return y0, y1


def _score_table(panel, hist: list[dict], best_iter: int) -> None:
    panel.set_facecolor(C["bg"])
    panel.axis("off")

    frame = FancyBboxPatch(
        (0.04, 0.14), 0.92, 0.72,
        transform=panel.transAxes,
        boxstyle="round,pad=0.008,rounding_size=0.015",
        facecolor="#FFFFFF",
        edgecolor=C["grid"],
        linewidth=1.4,
        zorder=0,
    )
    panel.add_patch(frame)

    col_labels = ["Iter", "ITM", "logic", "Overall"]
    rows = [
        [
            str(h["iteration"]),
            f"{h['image_text_match']:.2f}",
            f"{h['logic_score']:.2f}",
            f"{h['overall']:.3f}",
        ]
        for h in hist
    ]

    table = panel.table(
        cellText=rows,
        colLabels=col_labels,
        loc="center",
        bbox=[0.10, 0.22, 0.80, 0.56],
        cellLoc="center",
        edges="closed",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.15, 1.65)
    table.set_zorder(2)

    col_colors = [C["muted"], C["itm"], C["logic"], C["overall"]]
    for j in range(4):
        cell = table[(0, j)]
        cell.set_facecolor(C["table_head"])
        cell.set_text_props(fontweight="bold", color=C["text"])
        cell.set_edgecolor(C["grid"])
        cell.set_alpha(1.0)

    for i, h in enumerate(hist, start=1):
        is_best = h["iteration"] == best_iter
        for j in range(4):
            cell = table[(i, j)]
            cell.set_edgecolor(C["grid"])
            cell.set_facecolor(C["table_best"] if is_best else "#FFFFFF")
            cell.set_alpha(1.0)
            if j == 0:
                cell.set_text_props(fontweight="bold", color=C["text"])
            elif j == 3 and is_best:
                cell.set_text_props(fontweight="bold", color=C["overall"])
            else:
                cell.set_text_props(color=col_colors[j] if j > 0 else C["text"])


def draw_chart(out: Path = OUT) -> Path:
    hist = load_history()
    iters = [h["iteration"] for h in hist]
    overall = [h["overall"] for h in hist]
    itm = [h["image_text_match"] for h in hist]
    logic = [h["logic_score"] for h in hist]
    n = max(iters)
    y0, y1 = _y_limits(overall, itm, logic)

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig = plt.figure(figsize=(11, 5.5), dpi=150)
    fig.patch.set_facecolor(C["bg"])
    gs = fig.add_gridspec(
        1, 2, width_ratios=[1.7, 0.9], wspace=0.04,
        left=0.07, right=0.98, top=0.86, bottom=0.13,
    )
    ax = fig.add_subplot(gs[0, 0])
    tbl_ax = fig.add_subplot(gs[0, 1])
    ax.set_facecolor(C["bg"])

    ax.set_xlim(0.65, n + 0.35)
    ax.set_ylim(y0, y1)
    ax.set_xticks(iters)
    ax.set_xticklabels([f"Iter {i}" for i in iters], fontsize=11)
    ax.yaxis.set_major_locator(MultipleLocator(0.05 if (y1 - y0) <= 0.35 else 0.1))
    ax.set_ylabel("Score", fontsize=11, color=C["text"])
    ax.tick_params(colors=C["muted"], labelsize=10)
    ax.yaxis.grid(True, color=C["grid"], lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(C["grid"])

    series = [
        ("Overall", overall, C["overall"], "o"),
        ("ITM (image-text match)", itm, C["itm"], "s"),
        ("logic_score", logic, C["logic"], "^"),
    ]
    for label, vals, color, marker in series:
        ax.plot(iters, vals, color=color, marker=marker, markersize=9, lw=2.4,
                label=label, zorder=3, clip_on=True)

    ax.axhline(TAU, color=C["tau"], ls="--", lw=1.8, alpha=0.85, zorder=2)
    tau_label_y = TAU + (y1 - y0) * 0.025
    if tau_label_y > y1 - (y1 - y0) * 0.02:
        tau_label_y = TAU - (y1 - y0) * 0.045
    ax.text(n + 0.28, tau_label_y, f"τ = {TAU}", va="center", ha="right",
            fontsize=10, color=C["tau"], fontweight="bold", zorder=4)

    best_iter = max(iters, key=lambda i: overall[iters.index(i)])
    best_val = overall[iters.index(best_iter)]
    if best_val >= TAU:
        note = f"早停 @ iter {best_iter}  ·  Overall = {best_val:.3f}"
    else:
        note = f"保留最优 @ iter {best_iter}  ·  Overall = {best_val:.3f}"
    ax.text(
        0.98, 0.98, note,
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, fontweight="bold", color=C["tau"],
        bbox=dict(boxstyle="round,pad=0.4", fc="#FEF3C7", ec=C["tau"], alpha=0.95),
        zorder=5,
    )

    fig.suptitle(
        f"GRASP · 多指标迭代收敛（{VERSION}）",
        fontsize=14, fontweight="bold", color=C["text"], y=0.97,
    )
    fig.text(
        0.5, 0.03,
        "overall = 0.4×semantic + 0.35×layout_balance + 0.25×ITM  ·  max 5 iter · keep best",
        ha="center", fontsize=9, color=C["muted"],
    )

    leg = ax.legend(loc="upper left", frameon=True, fontsize=9.5, edgecolor=C["grid"])
    leg.get_frame().set_facecolor(C["bg"])
    leg.get_frame().set_alpha(0.95)

    _score_table(tbl_ax, hist, best_iter)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", facecolor=C["bg"])
    plt.close(fig)
    return out


def main() -> None:
    path = draw_chart()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
