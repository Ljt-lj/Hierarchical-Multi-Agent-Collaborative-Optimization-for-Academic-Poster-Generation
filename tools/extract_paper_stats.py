"""Extract experiment statistics for the ICLR report from saved pipeline outputs."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "Integrating_diverse_experimental_information_to_assist_prote"
IMAGES = ROOT / "s41592-025-02820-1_images"


def load_scores(name: str) -> list[dict]:
    path = OUT / f"{name}_scores.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("history", [])


def main() -> None:
    versions = ["v8", "v10", "v14", "v16"]
    print("=== Version comparison (best overall) ===")
    for v in versions:
        matches = list(OUT.glob(f"*{v}_scores.json"))
        if not matches:
            continue
        hist = json.loads(matches[0].read_text(encoding="utf-8")).get("history", [])
        best = max(h["overall"] for h in hist)
        print(f"{matches[0].stem:40s}  best={best:.3f}  iters={len(hist)}")

    print("\n=== v16 iteration detail ===")
    for h in load_scores("s41592_02820_v16"):
        bm = h.get("balance_metrics", {})
        print(
            f"iter={h['iteration']}  overall={h['overall']:.3f}  "
            f"logic={h['logic_score']:.2f}  LB={h['layout_balance']:.3f}  "
            f"ITM={h['image_text_match']:.2f}  SC={h['semantic_completeness']:.2f}  "
            f"overflow={bm.get('overflow', 0):.2f}  render_issues={h.get('render_issues', 0)}"
        )

    print("\n=== Figure crop vs focus ===")
    pairs = [
        ("fig1_focus.png", "fig1_crop_intro.png"),
        ("fig2_focus.png", "fig2_crop_benchmark.png"),
        ("fig3_focus.png", "fig3_crop_structure.png"),
    ]
    for focus_name, crop_name in pairs:
        fp, cp = IMAGES / focus_name, IMAGES / crop_name
        if not fp.exists() or not cp.exists():
            continue
        fw, fh = Image.open(fp).size
        cw, ch = Image.open(cp).size
        print(
            f"{crop_name:28s}  {cw}x{ch}  "
            f"area_ratio={cw * ch / (fw * fh):.3f}  "
            f"h_ratio={ch / fh:.3f}"
        )


if __name__ == "__main__":
    main()
