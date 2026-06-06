#!/usr/bin/env python3
"""从 Hugging Face 下载学术海报微调数据集."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_CACHE = TRAIN_ROOT / "data" / "raw"


def download_p2p_instruct(cache_dir: Path, max_rows: int | None = None) -> Path:
    from datasets import load_dataset

    print("下载 P2PInstruct (ASC8384/P2PInstruct) ...")
    split = f"train[:{max_rows}]" if max_rows else "train"
    ds = load_dataset("ASC8384/P2PInstruct", split=split, cache_dir=str(cache_dir))
    out = cache_dir / "p2p_instruct"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  保存 {len(ds)} 条 -> {path}")
    return path


def download_p2p_eval(cache_dir: Path) -> Path:
    from datasets import load_dataset

    print("下载 P2PEval (ASC8384/P2PEval) ...")
    ds = load_dataset("ASC8384/P2PEval", split="train", cache_dir=str(cache_dir))
    out = cache_dir / "p2p_eval"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "eval.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            rec = dict(row)
            if "image" in rec and rec["image"] is not None:
                img = rec["image"]
                if isinstance(img, dict) and "path" in img:
                    rec["image_path"] = img["path"]
                del rec["image"]
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    print(f"  保存 {len(ds)} 条 -> {path}")
    return path


def download_poster_sum(cache_dir: Path, split: str = "train", max_rows: int | None = None) -> Path:
    from datasets import load_dataset

    print(f"下载 PosterSum (rohitsaxena/PosterSum) split={split} ...")
    name = f"{split}[:{max_rows}]" if max_rows else split
    ds = load_dataset("rohitsaxena/PosterSum", split=name, cache_dir=str(cache_dir))
    out = cache_dir / "poster_sum"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{split}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    print(f"  保存 {len(ds)} 条 -> {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description="下载海报生成微调数据集")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset", choices=["all", "p2p_instruct", "p2p_eval", "poster_sum"], default="all")
    parser.add_argument("--max-rows", type=int, default=None, help="调试时限制条数")
    args = parser.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if args.dataset in ("all", "p2p_instruct"):
        paths.append(download_p2p_instruct(args.cache_dir, args.max_rows))
    if args.dataset in ("all", "p2p_eval"):
        paths.append(download_p2p_eval(args.cache_dir))
    if args.dataset in ("all", "poster_sum"):
        paths.append(download_poster_sum(args.cache_dir, "train", args.max_rows))
        paths.append(download_poster_sum(args.cache_dir, "validation", args.max_rows))

    manifest = {"files": [str(p) for p in paths], "cache_dir": str(args.cache_dir)}
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n完成。清单: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
