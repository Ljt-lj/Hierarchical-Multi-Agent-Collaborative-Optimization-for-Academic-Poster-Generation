#!/usr/bin/env python3
"""下载基座模型到本地，支持 ModelScope（国内云）与 HF 镜像."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

TRAIN_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = TRAIN_ROOT / "models"

MODELS = {
    "qwen2.5-7b": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "ms_id": "Qwen/Qwen2.5-7B-Instruct",
        "dir": "Qwen2.5-7B-Instruct",
        "size_hint": "~15GB",
    },
    "qwen2.5-1.5b": {
        "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "ms_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "dir": "Qwen2.5-1.5B-Instruct",
        "size_hint": "~3GB",
    },
}


def download_modelscope(model_id: str, local_dir: Path) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("安装 ModelScope: pip install modelscope")
        raise
    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"ModelScope 下载: {model_id} -> {local_dir}")
    path = snapshot_download(model_id, local_dir=str(local_dir))
    print(f"完成: {path}")
    return Path(path)


def download_hf(model_id: str, local_dir: Path, endpoint: str | None) -> Path:
    if endpoint:
        os.environ["HF_ENDPOINT"] = endpoint
        os.environ["HUGGINGFACE_HUB_ENDPOINT"] = endpoint
    local_dir.mkdir(parents=True, exist_ok=True)
    ep = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    print(f"HF 下载: {model_id} -> {local_dir} (endpoint={ep})")

    try:
        from huggingface_hub import snapshot_download
        path = snapshot_download(
            repo_id=model_id,
            local_dir=str(local_dir),
            resume_download=True,
            max_workers=4,
        )
        print(f"完成: {path}")
        return Path(path)
    except Exception as exc:
        print(f"huggingface_hub 失败: {exc}")
        print("尝试 huggingface-cli ...")
        cmd = [
            "huggingface-cli", "download", model_id,
            "--local-dir", str(local_dir),
            "--resume-download",
        ]
        subprocess.run(cmd, check=True)
        return local_dir


def verify_model_dir(path: Path) -> bool:
    if not path.exists():
        return False
    has_config = (path / "config.json").exists()
    has_tokenizer = (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()
    weights = list(path.glob("*.safetensors")) + list(path.glob("pytorch_model*.bin"))
    if has_config and has_tokenizer and weights:
        print(f"[OK] 模型完整: {len(weights)} 个权重文件")
        return True
    print(f"[WARN] 模型可能不完整: config={has_config} tokenizer={has_tokenizer} weights={len(weights)}")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description="下载 LoRA 基座模型")
    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()),
        default="qwen2.5-7b",
        help="qwen2.5-1.5b 更小，适合网络差或显存小",
    )
    parser.add_argument(
        "--source",
        choices=["modelscope", "hf", "auto"],
        default="auto",
        help="auto=国内优先 ModelScope",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--hf-endpoint", default="https://hf-mirror.com")
    args = parser.parse_args()

    spec = MODELS[args.model]
    local_dir = args.output_dir / spec["dir"]
    print(f"目标: {local_dir} ({spec['size_hint']})")

    if verify_model_dir(local_dir):
        print("已存在完整模型，跳过下载。")
        print(f"\n训练时使用:\n  python training/train/train_lora.py --model-path {local_dir} ...")
        return 0

    source = args.source
    if source == "auto":
        source = "modelscope"

    try:
        if source == "modelscope":
            download_modelscope(spec["ms_id"], local_dir)
        else:
            download_hf(spec["hf_id"], local_dir, args.hf_endpoint)
    except Exception as exc:
        print(f"\n[{source}] 下载失败: {exc}")
        if source == "modelscope":
            print("改用 HF 镜像重试 ...")
            download_hf(spec["hf_id"], local_dir, args.hf_endpoint)
        else:
            print("改用 ModelScope 重试: pip install modelscope && --source modelscope")
            return 1

    ok = verify_model_dir(local_dir)
    if ok:
        print(f"\n训练命令:\n  python training/train/train_lora.py \\")
        print(f"    --config training/configs/train_refiner_rocm.yaml \\")
        print(f"    --model-path {local_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
