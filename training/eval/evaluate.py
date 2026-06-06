#!/usr/bin/env python3
"""在验证集上评估微调模型（与 pipeline 相同的 JSON 解析逻辑）."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
ROOT = TRAIN_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poster_agent.llm_client import _parse_json
from training.train.dataset import load_jsonl


def _strict_json(text: str) -> bool:
    return text.strip().startswith("{")


def _parseable_json(text: str) -> bool:
    try:
        _parse_json(text)
        return True
    except (json.JSONDecodeError, ValueError):
        return False


def _load_model(base_model: str, adapter: Path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = dict(device_map="auto", trust_remote_code=True)
    dtype = torch.float16
    try:
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, **load_kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            base_model, torch_dtype=dtype, **load_kwargs
        )
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return tokenizer, model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA 输出目录 (final/)")
    parser.add_argument("--val-file", type=Path, default=TRAIN_ROOT / "data/processed/refiner_val.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    args = parser.parse_args()

    if not args.val_file.exists():
        print(f"验证集不存在: {args.val_file}")
        return 1

    rows = load_jsonl(args.val_file)[: args.max_samples]
    tokenizer, model = _load_model(args.base_model, args.adapter)

    gen_sig = inspect.signature(model.generate)
    gen_kwargs = dict(max_new_tokens=args.max_new_tokens, do_sample=False)
    if "pad_token_id" in gen_sig.parameters:
        gen_kwargs["pad_token_id"] = tokenizer.pad_token_id

    results: list[dict] = []
    for i, row in enumerate(rows):
        messages = row.get("messages", [])
        if len(messages) < 3:
            continue
        prompt_msgs = messages[:2]
        gold = messages[2]["content"]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        import torch

        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        results.append(
            {
                "idx": i,
                "strict_json": _strict_json(pred),
                "parseable_json": _parseable_json(pred),
                "gold_strict_json": _strict_json(gold),
                "gold_parseable_json": _parseable_json(gold),
                "pred_len": len(pred),
                "gold_len": len(gold),
                "pred_preview": pred[:400],
                "gold_preview": gold[:400],
            }
        )

    n = max(len(results), 1)
    report = {
        "samples": len(results),
        "strict_json_rate": sum(1 for r in results if r["strict_json"]) / n,
        "parseable_json_rate": sum(1 for r in results if r["parseable_json"]) / n,
        "gold_strict_json_rate": sum(1 for r in results if r["gold_strict_json"]) / n,
        "gold_parseable_json_rate": sum(1 for r in results if r["gold_parseable_json"]) / n,
        "note": (
            "strict_json=输出以{开头; parseable_json=与 pipeline 相同解析逻辑可解析。"
            "若 gold_strict_json_rate 也很低，说明验证集标签本身是 Markdown，需重建数据后重训。"
        ),
        "details": results[:10],
    }
    out_path = args.adapter / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
