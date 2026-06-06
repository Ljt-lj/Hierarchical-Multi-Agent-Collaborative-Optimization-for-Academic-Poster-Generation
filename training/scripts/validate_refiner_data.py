#!/usr/bin/env python3
"""校验 refiner_train/val.jsonl 标签能否被 pipeline 解析."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poster_agent.refiner_json import is_parseable_content_json, load_content_node_from_text, parse_error


def check_file(path: Path) -> int:
    if not path.exists():
        print(f"缺失: {path}")
        return 1
    ok = 0
    total = 0
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        total += 1
        gold = json.loads(line)["messages"][2]["content"]
        if is_parseable_content_json(gold):
            ok += 1
        else:
            err = parse_error(gold)
            print(f"[FAIL] {path.name} line {i}: {err}")
            print(f"       preview: {gold[:160]!r}")
    print(f"{path.name}: {ok}/{total} 可解析")
    return 0 if ok == total and total > 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "processed",
    )
    args = parser.parse_args()
    code = 0
    for name in ("refiner_train.jsonl", "refiner_val.jsonl"):
        path = args.data_dir / name
        if path.exists():
            code |= check_file(path)
        else:
            print(f"跳过缺失: {path}")
            code |= 1
    if code:
        print("\n修复: git pull && python training/data/build_sft.py")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
