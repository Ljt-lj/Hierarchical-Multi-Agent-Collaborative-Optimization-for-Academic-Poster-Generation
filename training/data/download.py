#!/usr/bin/env python3
"""从 Hugging Face 下载学术海报微调数据集."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data.p2p_classify import is_figure_description_task, is_refiner_content, messages_to_pair

DEFAULT_CACHE = TRAIN_ROOT / "data" / "raw"

# 训练 build_sft.py 至少需要 p2p_instruct；其余为评测/增强，失败可跳过
REQUIRED_FOR_TRAIN = {"p2p_instruct"}
OPTIONAL_DATASETS = {"p2p_eval", "poster_sum"}


def configure_hf(endpoint: str | None = None) -> str:
    """设置 HF 镜像/端点，返回实际使用的 endpoint."""
    ep = endpoint or os.environ.get("HF_ENDPOINT") or os.environ.get("HUGGINGFACE_HUB_ENDPOINT") or ""
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    if ep:
        print(f"HF 端点: {ep}")
    else:
        print("HF 端点: https://huggingface.co （国内云建议: export HF_ENDPOINT=https://hf-mirror.com）")
    return ep or "https://huggingface.co"


def download_p2p_instruct(
    cache_dir: Path,
    max_rows: int | None = None,
    *,
    refiner_quota: int | None = None,
    visual_quota: int | None = None,
) -> Path:
    from datasets import load_dataset

    out = cache_dir / "p2p_instruct"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "train.jsonl"

    if refiner_quota or visual_quota:
        rq = refiner_quota or 0
        vq = visual_quota or 0
        print(
            "下载 P2PInstruct ... [按任务均衡抽样]"
            f" refiner_quota={rq} visual_quota={vq}"
        )
        print("  说明: 数据集前段几乎全是 figure descriptor，不可直接用 train[:500]")
        ds = load_dataset("ASC8384/P2PInstruct", split="train", streaming=True, cache_dir=str(cache_dir))
        refiner_rows: list[dict] = []
        visual_rows: list[dict] = []
        scanned = 0
        for row in ds:
            scanned += 1
            messages = row.get("messages") or []
            _, user, assistant = messages_to_pair(messages)
            if rq and len(refiner_rows) < rq and is_refiner_content(user, assistant):
                refiner_rows.append(dict(row))
            elif vq and len(visual_rows) < vq and is_figure_description_task(user, assistant):
                visual_rows.append(dict(row))
            if (not rq or len(refiner_rows) >= rq) and (not vq or len(visual_rows) >= vq):
                break
            if scanned % 5000 == 0:
                print(f"  已扫描 {scanned} 条 | refiner={len(refiner_rows)} visual={len(visual_rows)}")
        rows = refiner_rows + visual_rows
        if rq and len(refiner_rows) < rq:
            print(f"  [警告] refiner 仅凑够 {len(refiner_rows)}/{rq}")
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"  扫描 {scanned} 条，写入 {len(rows)} 条 -> {path}")
        return path

    print("下载 P2PInstruct (ASC8384/P2PInstruct) ... [训练必需]")
    if max_rows:
        print(
            "  [警告] train[:N] 前段多为 figure descriptor，Refiner 训练请改用 "
            "--refiner-quota 500 --visual-quota 200"
        )
    split = f"train[:{max_rows}]" if max_rows else "train"
    ds = load_dataset("ASC8384/P2PInstruct", split=split, cache_dir=str(cache_dir))
    with path.open("w", encoding="utf-8") as f:
        for row in ds:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  保存 {len(ds)} 条 -> {path}")
    return path


def download_p2p_eval(cache_dir: Path) -> Path:
    from datasets import load_dataset

    print("下载 P2PEval (ASC8384/P2PEval) ... [评测可选]")
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

    print(f"下载 PosterSum (rohitsaxena/PosterSum) split={split} ... [增强可选]")
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


DOWNLOADERS = {
    "p2p_instruct": lambda cache, max_rows, refiner_quota, visual_quota: download_p2p_instruct(
        cache, max_rows, refiner_quota=refiner_quota, visual_quota=visual_quota
    ),
    "p2p_eval": lambda cache, max_rows, refiner_quota, visual_quota: download_p2p_eval(cache),
    "poster_sum": lambda cache, max_rows, refiner_quota, visual_quota: [
        download_poster_sum(cache, "train", max_rows),
        download_poster_sum(cache, "validation", max_rows),
    ],
}


def _run_one(
    name: str,
    cache_dir: Path,
    max_rows: int | None,
    refiner_quota: int | None,
    visual_quota: int | None,
) -> list[Path]:
    fn = DOWNLOADERS[name]
    result = fn(cache_dir, max_rows, refiner_quota, visual_quota)
    if isinstance(result, list):
        return result
    return [result]


def main() -> int:
    parser = argparse.ArgumentParser(description="下载海报生成微调数据集")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument(
        "--dataset",
        choices=["all", "p2p_instruct", "p2p_eval", "poster_sum", "train_minimal"],
        default="all",
        help="train_minimal = 仅 p2p_instruct（训练最小集）",
    )
    parser.add_argument("--max-rows", type=int, default=None, help="顺序截取前 N 条（不推荐用于 Refiner）")
    parser.add_argument(
        "--refiner-quota",
        type=int,
        default=None,
        help="扫描全集直到凑够 N 条章节/内容生成样本（推荐）",
    )
    parser.add_argument(
        "--visual-quota",
        type=int,
        default=None,
        help="扫描全集直到凑够 N 条 figure descriptor 样本",
    )
    parser.add_argument("--hf-endpoint", type=str, default=None, help="如 https://hf-mirror.com")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=True,
        help="可选数据集下载失败时继续（默认开启）",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="任一数据集失败则整体退出（关闭 skip-failed）",
    )
    args = parser.parse_args()

    if args.strict:
        args.skip_failed = False

    configure_hf(args.hf_endpoint)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "train_minimal":
        targets = ["p2p_instruct"]
        if args.refiner_quota is None and args.visual_quota is None and args.max_rows is None:
            args.refiner_quota = 400
            args.visual_quota = 100
            print("train_minimal 默认: --refiner-quota 400 --visual-quota 100")
    elif args.dataset == "all":
        targets = ["p2p_instruct", "p2p_eval", "poster_sum"]
    else:
        targets = [args.dataset]

    paths: list[Path] = []
    errors: list[str] = []

    for name in targets:
        required = name in REQUIRED_FOR_TRAIN
        try:
            got = _run_one(name, args.cache_dir, args.max_rows, args.refiner_quota, args.visual_quota)
            paths.extend(got)
        except Exception as exc:
            msg = f"{name}: {exc}"
            errors.append(msg)
            print(f"\n[失败] {name}")
            traceback.print_exc()
            if required or not args.skip_failed:
                print(f"\n必需数据集 {name} 下载失败，退出。")
                return 1
            print(f"[跳过] {name} 为可选数据集，继续下一个 ...")

    manifest = {
        "files": [str(p) for p in paths],
        "cache_dir": str(args.cache_dir),
        "errors": errors,
        "hf_endpoint": os.environ.get("HF_ENDPOINT", ""),
    }
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n完成。成功 {len(paths)} 个文件，失败 {len(errors)} 个。")
    print(f"清单: {manifest_path}")
    if paths and (args.cache_dir / "p2p_instruct" / "train.jsonl").exists():
        print("\n可继续训练:")
        print("  python training/data/build_sft.py")
        print("  python training/train/train_lora.py --config training/configs/train_refiner_rocm.yaml")
    if errors:
        print("\n可选数据集稍后重试:")
        print("  export HF_ENDPOINT=https://hf-mirror.com")
        print("  python training/data/download.py --dataset p2p_eval")
        print("  python training/data/download.py --dataset poster_sum")
    return 0 if (args.cache_dir / "p2p_instruct" / "train.jsonl").exists() else 1


if __name__ == "__main__":
    raise SystemExit(main())
