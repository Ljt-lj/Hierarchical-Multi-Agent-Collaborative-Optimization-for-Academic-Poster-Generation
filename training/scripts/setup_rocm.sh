#!/usr/bin/env bash
# AMD ROCm 云服务器一键环境检查 + 依赖安装（在云端 Linux 执行）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

echo "== ROCm 驱动 =="
if command -v rocm-smi &>/dev/null; then rocm-smi --showproductname || rocm-smi; fi
if command -v amd-smi &>/dev/null; then amd-smi version || true; fi

echo "== Python =="
python3 --version

echo "== 安装依赖（不含 bitsandbytes，AMD 不兼容）=="
pip install -r requirements.txt
pip install -r training/requirements.txt || true
pip uninstall bitsandbytes -y 2>/dev/null || true

echo "== 若 GPU 不可用，安装 ROCm 版 PyTorch =="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda_ok", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY

echo "== GPU 检查 =="
python3 training/scripts/check_gpu.py || true

echo "完成。训练示例:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo "  python training/data/download.py --max-rows 500"
echo "  python training/data/build_sft.py"
echo "  python training/train/train_lora.py --config training/configs/train_refiner_rocm.yaml"
