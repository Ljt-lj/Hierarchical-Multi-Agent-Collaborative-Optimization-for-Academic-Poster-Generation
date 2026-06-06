#!/usr/bin/env python3
"""将原始 HF 数据集转换为各 Agent 的 SFT JSONL."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRAIN_ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.data.schemas import (
    COMMENTER_SYSTEM,
    REFINER_SYSTEM,
    VISUAL_SYSTEM,
    ChatMessage,
    SFTSample,
    TaskType,
)

RAW_DIR = TRAIN_ROOT / "data" / "raw"
OUT_DIR = TRAIN_ROOT / "data" / "processed"


def _write_jsonl(path: Path, samples: list[SFTSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def _classify_p2p_task(user_text: str, assistant_text: str) -> TaskType:
    u, a = user_text.lower(), assistant_text.lower()
    if "figure" in u or "image" in u or "caption" in u or "visual" in u:
        return TaskType.VISUAL
    if "html" in u or "css" in u or "flexbox" in u or "layout" in u and "column" in u:
        return TaskType.ORCHESTRATOR
    if "evaluate" in u or "score" in u or "judge" in u:
        return TaskType.COMMENTER
    if "section" in u or "markdown" in a or "abstract" in u or "introduction" in u:
        return TaskType.REFINER
    if any(k in a for k in ("bar_chart", "line_chart", "flow", "architecture", "stat")):
        return TaskType.VISUAL
    if len(assistant_text) > 800:
        return TaskType.REFINER
    return TaskType.VISUAL


def _messages_to_pair(messages: list[dict]) -> tuple[str, str, str]:
    system = ""
    user_parts: list[str] = []
    assistant = ""
    for m in messages:
        role = m.get("role", "")
        content = (m.get("content") or "").strip()
        if role == "system":
            system = content
        elif role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant = content
    return system, "\n\n".join(user_parts), assistant


def build_from_p2p_instruct(raw_path: Path) -> dict[str, list[SFTSample]]:
    buckets: dict[str, list[SFTSample]] = {
        TaskType.REFINER.value: [],
        TaskType.VISUAL.value: [],
        TaskType.COMMENTER.value: [],
        TaskType.ORCHESTRATOR.value: [],
    }
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            _, user, assistant = _messages_to_pair(messages)
            if not user or not assistant:
                continue
            task = _classify_p2p_task(user, assistant)
            if task == TaskType.ORCHESTRATOR:
                continue
            system_map = {
                TaskType.REFINER: REFINER_SYSTEM,
                TaskType.VISUAL: VISUAL_SYSTEM,
                TaskType.COMMENTER: COMMENTER_SYSTEM,
            }
            sample = SFTSample(
                task=task.value,
                messages=[
                    ChatMessage("system", system_map.get(task, REFINER_SYSTEM)),
                    ChatMessage("user", user[:12000]),
                    ChatMessage("assistant", assistant[:12000]),
                ],
                meta={"source": "p2p_instruct"},
            )
            buckets[task.value].append(sample)
    return buckets


def build_from_poster_sum(raw_path: Path) -> list[SFTSample]:
    """摘要 → 海报内容树 JSON，用于 Refiner 数据增强."""
    samples: list[SFTSample] = []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            title = row.get("title", "")
            abstract = row.get("abstract", "")
            topics = row.get("topics") or []
            if not abstract:
                continue
            bullets = [abstract[:200]]
            if topics:
                bullets.append(f"Topics: {', '.join(topics[:4])}")
            output = {
                "title": title,
                "summary": abstract[:120],
                "bullets": bullets[:3],
                "weight": 0.9,
                "logic_links": [],
                "children": [
                    {
                        "title": "Abstract",
                        "summary": abstract[:120],
                        "bullets": bullets[:3],
                        "weight": 0.1,
                        "logic_links": [],
                        "children": [],
                    }
                ],
            }
            user = f"Paper title: {title}\nAbstract:\n{abstract}"
            samples.append(
                SFTSample(
                    task=TaskType.REFINER.value,
                    messages=[
                        ChatMessage("system", REFINER_SYSTEM),
                        ChatMessage("user", user),
                        ChatMessage("assistant", json.dumps(output, ensure_ascii=False)),
                    ],
                    meta={"source": "poster_sum", "conference": row.get("conference", "")},
                )
            )
    return samples


def split_train_val(samples: list[SFTSample], val_ratio: float = 0.05, seed: int = 42) -> tuple[list, list]:
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_ratio))
    return shuffled[n_val:], shuffled[:n_val]


def main() -> int:
    parser = argparse.ArgumentParser(description="构建 Agent SFT 数据集")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    p2p_path = args.raw_dir / "p2p_instruct" / "train.jsonl"
    poster_path = args.raw_dir / "poster_sum" / "train.jsonl"

    if not p2p_path.exists():
        print(f"缺少 {p2p_path}，请先运行: python training/data/download.py")
        return 1

    buckets = build_from_p2p_instruct(p2p_path)
    if poster_path.exists():
        augment = build_from_poster_sum(poster_path)
        buckets[TaskType.REFINER.value].extend(augment[: min(3000, len(augment))])
        print(f"PosterSum 增强 Refiner: +{min(3000, len(augment))} 条")

    stats: dict[str, dict] = {}
    for task, samples in buckets.items():
        if not samples:
            continue
        train, val = split_train_val(samples, args.val_ratio, args.seed)
        _write_jsonl(args.out_dir / f"{task}_train.jsonl", train)
        _write_jsonl(args.out_dir / f"{task}_val.jsonl", val)
        stats[task] = {"train": len(train), "val": len(val)}
        print(f"{task}: train={len(train)} val={len(val)}")

    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计: {stats_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
