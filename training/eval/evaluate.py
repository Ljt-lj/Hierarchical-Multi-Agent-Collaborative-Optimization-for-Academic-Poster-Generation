#!/usr/bin/env python3
"""在验证集上评估微调模型."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
ROOT = TRAIN_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from training.train.dataset import load_jsonl


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True, help="LoRA 输出目录 (final/)")
    parser.add_argument("--val-file", type=Path, default=TRAIN_ROOT / "data/processed/refiner_val.jsonl")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-samples", type=int, default=50)
    args = parser.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = load_jsonl(args.val_file)[: args.max_samples]
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    results: list[dict] = []
    for i, row in enumerate(rows):
        messages = row.get("messages", [])
        if len(messages) < 3:
            continue
        prompt_msgs = messages[:2]
        gold = messages[2]["content"]
        prompt = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
        pred = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        ok_json = pred.strip().startswith("{")
        results.append({"idx": i, "valid_json": ok_json, "pred_len": len(pred), "gold_len": len(gold)})

    valid = sum(1 for r in results if r["valid_json"])
    report = {
        "samples": len(results),
        "valid_json_rate": valid / max(len(results), 1),
        "details": results[:10],
    }
    out_path = args.adapter / "eval_report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"报告: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
