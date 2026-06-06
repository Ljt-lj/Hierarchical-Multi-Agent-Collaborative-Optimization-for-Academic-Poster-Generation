#!/usr/bin/env python3
"""LoRA 监督微调：Refiner / Visual Agent."""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import yaml

TRAIN_ROOT = Path(__file__).resolve().parents[1]
ROOT = TRAIN_ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base: Path, p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else base / path


def main() -> int:
    parser = argparse.ArgumentParser(description="LoRA 微调海报 Agent")
    parser.add_argument("--config", type=Path, default=TRAIN_ROOT / "configs" / "train_refiner.yaml")
    parser.add_argument("--max-samples", type=int, default=None, help="调试：限制训练样本数")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_file = resolve_path(TRAIN_ROOT, cfg["train_file"])
    val_file = resolve_path(TRAIN_ROOT, cfg["val_file"])
    output_dir = resolve_path(TRAIN_ROOT, cfg["output_dir"])

    if not train_file.exists():
        print(f"训练集不存在: {train_file}")
        print("请先: python training/data/download.py && python training/data/build_sft.py")
        return 1

    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        print("[错误] 未检测到可用 GPU（CUDA/ROCm）。")
        print("请先运行: python training/scripts/check_gpu.py")
        print("AMD 云 GPU 见 SHELL.md「AMD ROCm 云端」章节")
        return 1

    from training.train.dataset import build_hf_dataset

    is_rocm = bool(getattr(torch.version, "hip", None))
    use_4bit = bool(cfg.get("load_in_4bit", True)) and not is_rocm and platform.system() != "Windows"
    if is_rocm and cfg.get("load_in_4bit"):
        print("[提示] AMD ROCm 不支持 bitsandbytes 4bit，已自动使用 fp16 全精度 LoRA")
    if cfg.get("load_in_4bit") and platform.system() == "Windows":
        print("[提示] Windows 未启用 4bit 量化，使用 fp16/bf16 全精度加载（需更大显存）")

    model_name = cfg["base_model"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = None
    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        torch_dtype=torch.bfloat16 if cfg.get("bf16") else torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        r=int(cfg.get("lora_r", 16)),
        lora_alpha=int(cfg.get("lora_alpha", 32)),
        lora_dropout=float(cfg.get("lora_dropout", 0.05)),
        target_modules=list(cfg.get("target_modules", ["q_proj", "v_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_ds = build_hf_dataset(train_file, tokenizer, args.max_samples)
    eval_ds = build_hf_dataset(val_file, tokenizer, max(64, (args.max_samples or 500) // 10))

    def tokenize(batch):
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=int(cfg.get("max_seq_length", 4096)),
            padding="max_length",
        )
        out["labels"] = [ids[:] for ids in out["input_ids"]]
        return out

    train_ds = train_ds.map(tokenize, batched=True, remove_columns=["text"])
    eval_ds = eval_ds.map(tokenize, batched=True, remove_columns=["text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(cfg.get("per_device_train_batch_size", 2)),
        per_device_eval_batch_size=int(cfg.get("per_device_eval_batch_size", 2)),
        gradient_accumulation_steps=int(cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(cfg.get("learning_rate", 2e-4)),
        warmup_ratio=float(cfg.get("warmup_ratio", 0.03)),
        logging_steps=int(cfg.get("logging_steps", 10)),
        eval_strategy="steps",
        eval_steps=int(cfg.get("eval_steps", 200)),
        save_steps=int(cfg.get("save_steps", 400)),
        save_total_limit=int(cfg.get("save_total_limit", 3)),
        bf16=bool(cfg.get("bf16")),
        fp16=not bool(cfg.get("bf16")),
        report_to="none",
        seed=int(cfg.get("seed", 42)),
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\n微调完成: {output_dir / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
