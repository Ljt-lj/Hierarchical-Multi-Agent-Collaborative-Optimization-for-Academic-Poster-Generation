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

# P2PInstruct 两类样本长度差异（论文）：图表描述 ~192 token，章节生成 ~3300 token
REFINER_MIN_CHARS = 900
VISUAL_MAX_CHARS = 700

VISUAL_USER_MARKERS = (
    "figure descriptor",
    "describe the figure",
    "describe this figure",
    "describe the image",
    "visual element",
    "caption",
    "chart",
    "table",
    "figure caption",
    "image caption",
)

REFINER_USER_MARKERS = (
    "section generator",
    "content generator",
    "poster section",
    "poster content",
    "compress",
    "summarize",
    "content tree",
    "refine",
    "bullet point",
    "abstract",
    "introduction",
    "conclusion",
    "methodology",
    "experiment",
    "document tree",
)


def _is_figure_description_task(user: str, assistant: str) -> bool:
    u, a = user.lower(), assistant.lower()
    if any(m in u for m in VISUAL_USER_MARKERS):
        return True
    if any(m in u for m in ("figure", "image", "visual", "caption")) and len(assistant) <= VISUAL_MAX_CHARS * 3:
        return True
    opening = a[:280]
    if opening.startswith(("this figure", "the figure", "this image", "the image")):
        return True
    if "illustrates" in opening and ("figure" in opening or "image" in opening):
        return True
    return False


def _extract_bullets(text: str, limit: int = 4) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "•", "*")):
            bullets.append(stripped.lstrip("-•* ").strip())
        elif stripped and len(stripped) < 200 and stripped[0].isdigit() and "." in stripped[:4]:
            bullets.append(stripped)
    if bullets:
        return bullets[:limit]
    parts = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 10]
    return parts[:limit] if parts else [text[:120]]


def _markdown_sections_to_json(text: str) -> dict | None:
    import re

    sections = re.split(r"\n(?=##\s+)", text.strip())
    if len(sections) <= 1 and not text.strip().startswith("##"):
        return None

    children: list[dict] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        match = re.match(r"##\s+(.+?)(?:\n|$)", sec)
        if match:
            title = match.group(1).strip()
            body = sec[match.end() :].strip()
        else:
            title = "Section"
            body = sec
        bullets = _extract_bullets(body)
        children.append(
            {
                "title": title,
                "summary": body[:120],
                "bullets": bullets[:4],
                "weight": 0.5,
                "logic_links": [],
                "children": [],
            }
        )
    if not children:
        return None
    return {
        "title": children[0]["title"],
        "summary": children[0]["summary"],
        "bullets": children[0]["bullets"][:3],
        "weight": 1.0,
        "logic_links": [],
        "children": children[1:] if len(children) > 1 else [],
    }


def normalize_refiner_assistant(text: str, user: str = "") -> str | None:
    """将 P2P Markdown 章节转为 RefinerAgent 期望的 JSON 标签."""
    text = text.strip()
    if not text or _is_figure_description_task(user, text):
        return None

    if text.startswith("{") or text.startswith("["):
        try:
            obj = json.loads(text)
            return json.dumps(obj, ensure_ascii=False)
        except json.JSONDecodeError:
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                try:
                    obj = json.loads(text[start : end + 1])
                    return json.dumps(obj, ensure_ascii=False)
                except json.JSONDecodeError:
                    pass

    if "##" in text or re.search(r"^#+\s", text, re.MULTILINE):
        tree = _markdown_sections_to_json(text)
        if tree:
            return json.dumps(tree, ensure_ascii=False)

    u = user.lower()
    if len(text) >= REFINER_MIN_CHARS and any(m in u for m in REFINER_USER_MARKERS):
        return json.dumps(
            {
                "title": "Poster",
                "summary": text[:120],
                "bullets": _extract_bullets(text)[:4],
                "weight": 1.0,
                "logic_links": [],
                "children": [],
            },
            ensure_ascii=False,
        )
    return None


def _write_jsonl(path: Path, samples: list[SFTSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")


def _classify_p2p_task(user_text: str, assistant_text: str) -> TaskType:
    """P2PInstruct 任务分类：章节/内容生成 → refiner，图表描述 → visual."""
    u, a = user_text.lower(), assistant_text.lower()
    n = len(assistant_text)

    # HTML 组装 → 跳过
    if "html" in u and ("css" in u or "flexbox" in u or "render" in u):
        return TaskType.ORCHESTRATOR
    if "<html" in a or "<!doctype" in a:
        return TaskType.ORCHESTRATOR

    if "evaluate" in u or "judge" in u or ("score" in u and "poster" in u):
        return TaskType.COMMENTER

    if _is_figure_description_task(user_text, assistant_text):
        return TaskType.VISUAL

    # Refiner 强信号：结构化内容树 / 长 Markdown 章节
    refiner_assistant_markers = (
        '"children"', '"bullets"', '"logic_links"', '"weight"',
        "## abstract", "## introduction", "## method", "## experiment", "## conclusion",
    )
    if any(m in a for m in refiner_assistant_markers):
        return TaskType.REFINER
    if n >= REFINER_MIN_CHARS and ("##" in assistant_text or "###" in assistant_text):
        return TaskType.REFINER

    refiner_user_markers = REFINER_USER_MARKERS
    if any(m in u for m in refiner_user_markers):
        return TaskType.REFINER

    # 长回复默认为章节/内容生成（P2P 论文 ~3300 tokens），但排除图表描述
    if n >= 1800 and not _is_figure_description_task(user_text, assistant_text):
        return TaskType.REFINER
    if n >= REFINER_MIN_CHARS and ("##" in assistant_text or "###" in assistant_text):
        return TaskType.REFINER

    # Visual 强信号：图表/图片描述（~192 tokens）
    visual_user_markers = VISUAL_USER_MARKERS
    if any(m in u for m in visual_user_markers):
        return TaskType.VISUAL
    if n <= VISUAL_MAX_CHARS and any(m in u for m in ("figure", "image", "visual", "caption")):
        return TaskType.VISUAL

    if any(m in a for m in ("bar_chart", "line_chart", "stat_cards", "flow_diagram")):
        return TaskType.VISUAL

    # 中等长度：偏 refiner（poster 文本多于纯 caption）
    if n >= 500:
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
    counts = {"total": 0, "skip_orchestrator": 0, "skip_refiner_normalize": 0}
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            messages = row.get("messages") or []
            if len(messages) < 2:
                continue
            _, user, assistant = _messages_to_pair(messages)
            if not user or not assistant:
                continue
            counts["total"] += 1
            task = _classify_p2p_task(user, assistant)
            if task == TaskType.ORCHESTRATOR:
                counts["skip_orchestrator"] += 1
                continue
            system_map = {
                TaskType.REFINER: REFINER_SYSTEM,
                TaskType.VISUAL: VISUAL_SYSTEM,
                TaskType.COMMENTER: COMMENTER_SYSTEM,
            }
            assistant_out = assistant[:12000]
            if task == TaskType.REFINER:
                normalized = normalize_refiner_assistant(assistant, user)
                if normalized is None:
                    counts["skip_refiner_normalize"] += 1
                    continue
                assistant_out = normalized
            sample = SFTSample(
                task=task.value,
                messages=[
                    ChatMessage("system", system_map.get(task, REFINER_SYSTEM)),
                    ChatMessage("user", user[:12000]),
                    ChatMessage("assistant", assistant_out),
                ],
                meta={"source": "p2p_instruct"},
            )
            buckets[task.value].append(sample)
    print(
        f"  解析 {counts['total']} 条，跳过 orchestrator {counts['skip_orchestrator']} 条，"
        f"跳过无法转 JSON 的 refiner {counts['skip_refiner_normalize']} 条"
    )
    return buckets


def ensure_refiner_bucket(buckets: dict[str, list[SFTSample]], min_ratio: float = 0.25) -> None:
    """若 refiner 过少，仅从 visual 中挑选可转为 JSON 章节的长样本."""
    refiner = buckets[TaskType.REFINER.value]
    visual = buckets[TaskType.VISUAL.value]
    total = len(refiner) + len(visual)
    if total == 0:
        return
    if len(refiner) >= total * min_ratio:
        return

    need = max(1, int(total * min_ratio) - len(refiner))
    candidates: list[SFTSample] = []
    for s in visual:
        user = s.messages[1].content
        assistant = s.messages[2].content
        if _is_figure_description_task(user, assistant):
            continue
        if "##" not in assistant and len(assistant) < REFINER_MIN_CHARS:
            continue
        normalized = normalize_refiner_assistant(assistant, user)
        if not normalized:
            continue
        s.task = TaskType.REFINER.value
        s.messages[0] = ChatMessage("system", REFINER_SYSTEM)
        s.messages[2] = ChatMessage("assistant", normalized)
        s.meta["source"] = "p2p_instruct_refiner_fallback"
        candidates.append(s)

    candidates.sort(key=lambda s: len(s.messages[2].content), reverse=True)
    move = candidates[:need]
    moved_ids = {id(s) for s in move}
    buckets[TaskType.VISUAL.value] = [s for s in visual if id(s) not in moved_ids]
    buckets[TaskType.REFINER.value].extend(move)
    print(f"  [fallback] 从 visual 可转 JSON 章节样本补充 refiner +{len(move)} 条")
    if len(refiner) + len(move) < total * min_ratio:
        print(
            "  [警告] refiner 仍偏少，建议: python training/data/download.py --max-rows 5000 "
            "后重新 build_sft"
        )


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
    n_val = max(1, int(len(shuffled) * val_ratio)) if len(shuffled) > 1 else 0
    if n_val == 0:
        return shuffled, []
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
    ensure_refiner_bucket(buckets)

    if poster_path.exists():
        augment = build_from_poster_sum(poster_path)
        buckets[TaskType.REFINER.value].extend(augment[: min(3000, len(augment))])
        print(f"PosterSum 增强 Refiner: +{min(3000, len(augment))} 条")

    stats: dict[str, dict] = {}
    wrote_any = False
    for task, samples in buckets.items():
        if not samples:
            print(f"{task}: 0 条（跳过）")
            continue
        train, val = split_train_val(samples, args.val_ratio, args.seed)
        _write_jsonl(args.out_dir / f"{task}_train.jsonl", train)
        _write_jsonl(args.out_dir / f"{task}_val.jsonl", val)
        stats[task] = {"train": len(train), "val": len(val)}
        print(f"{task}: train={len(train)} val={len(val)}")
        wrote_any = True

    stats_path = args.out_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"统计: {stats_path}")

    refiner_val = args.out_dir / "refiner_val.jsonl"
    if refiner_val.exists():
        val_rows = refiner_val.read_text(encoding="utf-8").strip().splitlines()
        json_gold = 0
        for line in val_rows:
            row = json.loads(line)
            gold = row["messages"][2]["content"]
            if gold.strip().startswith("{"):
                json_gold += 1
        print(f"refiner_val JSON 标签率: {json_gold}/{len(val_rows)}")
        if val_rows and json_gold == 0:
            print("[错误] refiner 验证集标签仍非 JSON，请确认已同步最新 build_sft.py")
            return 1

    if not (args.out_dir / "refiner_train.jsonl").exists():
        print("\n[错误] 仍未生成 refiner_train.jsonl，请检查原始数据或增大 --max-rows")
        return 1
    return 0 if wrote_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
