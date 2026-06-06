"""SFT 数据集加载."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from datasets import Dataset


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_chat(messages: list[dict[str, str]], tokenizer) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    parts: list[str] = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}")
    return "\n\n".join(parts)


def build_hf_dataset(path: Path, tokenizer, max_samples: int | None = None) -> Dataset:
    rows = load_jsonl(path)
    if max_samples:
        rows = rows[:max_samples]

    messages_list: list[list[dict[str, str]]] = []
    texts: list[str] = []
    for row in rows:
        messages = row.get("messages") or []
        if len(messages) < 3:
            continue
        messages_list.append(messages)
        texts.append(format_chat(messages, tokenizer))

    return Dataset.from_dict({"text": texts, "messages": messages_list})
