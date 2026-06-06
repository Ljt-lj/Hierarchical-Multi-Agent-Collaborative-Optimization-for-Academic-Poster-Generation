#!/usr/bin/env python3
"""检查 GPU 环境：NVIDIA CUDA / AMD ROCm."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys


def _run_cmd(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def detect_amd_rocm() -> dict:
    info: dict = {"rocm_smi": None, "amd_smi": None, "lines": []}
    for name in ("rocm-smi", "amd-smi"):
        path = shutil.which(name)
        if path:
            info[name.replace("-", "_")] = path
            out = _run_cmd([name, "--showproductname"]) or _run_cmd([name])
            if out:
                info["lines"].extend(out.splitlines()[:5])
    return info


def detect_nvidia() -> dict:
    info: dict = {"nvidia_smi": None, "lines": []}
    path = shutil.which("nvidia-smi")
    if path:
        info["nvidia_smi"] = path
        out = _run_cmd([
            "nvidia-smi", "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ])
        if out:
            info["lines"] = out.splitlines()
    return info


def torch_backend() -> dict:
    out: dict = {"installed": False, "hip": False, "cuda_ok": False, "devices": []}
    try:
        import torch
    except ImportError:
        return out
    out["installed"] = True
    out["version"] = torch.__version__
    out["hip"] = bool(getattr(torch.version, "hip", None))
    out["cuda_ok"] = torch.cuda.is_available()
    if out["cuda_ok"]:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            vram = props.total_memory / (1024 ** 3)
            out["devices"].append({"index": i, "name": props.name, "vram_gb": round(vram, 2)})
    return out


def recommend_config(vram_gb: float, is_amd: bool) -> str:
    if is_amd:
        if vram_gb >= 24:
            return "training/configs/train_refiner_rocm.yaml"
        if vram_gb >= 12:
            return "training/configs/train_refiner_rocm.yaml  # 或 lowvram 若 OOM"
        return "training/configs/train_refiner_lowvram.yaml"
    if vram_gb >= 14:
        return "training/configs/train_refiner.yaml"
    if vram_gb >= 6:
        return "training/configs/train_refiner_lowvram.yaml"
    return "training/configs/train_refiner_lowvram.yaml  # 显存偏小，注意 OOM"


def main() -> int:
    print("=" * 55)
    print("GPU 环境检查 (NVIDIA CUDA / AMD ROCm)")
    print("=" * 55)
    print(f"系统: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")

    nvidia = detect_nvidia()
    amd = detect_amd_rocm()
    torch_info = torch_backend()

    is_nvidia = bool(nvidia["nvidia_smi"])
    is_amd = bool(amd.get("rocm_smi") or amd.get("amd_smi") or torch_info.get("hip"))

    if nvidia["nvidia_smi"]:
        print(f"\n[OK] NVIDIA: {nvidia['nvidia_smi']}")
        for line in nvidia["lines"]:
            print(f"  {line}")
    else:
        print("\n[INFO] 未检测到 nvidia-smi（非 NVIDIA 或未装驱动）")

    if amd.get("rocm_smi") or amd.get("amd_smi"):
        tool = amd.get("rocm_smi") or amd.get("amd_smi")
        print(f"\n[OK] AMD ROCm 工具: {tool}")
        for line in amd.get("lines", []):
            print(f"  {line}")
    elif is_amd:
        print("\n[OK] PyTorch 为 ROCm/HIP 构建")
    else:
        print("\n[INFO] 未检测到 rocm-smi / amd-smi")

    if torch_info["installed"]:
        print(f"\nPyTorch: {torch_info['version']}")
        print(f"ROCm/HIP 构建: {torch_info['hip']}")
        print(f"GPU 可用 (torch.cuda): {torch_info['cuda_ok']}")
        for d in torch_info["devices"]:
            print(f"  [{d['index']}] {d['name']} | {d['vram_gb']} GB")
        if torch_info["devices"]:
            vram = torch_info["devices"][0]["vram_gb"]
            cfg = recommend_config(vram, is_amd)
            print(f"\n[推荐配置] {cfg}")
            if is_amd:
                print("[AMD 注意] 请勿使用 4bit 量化 (bitsandbytes)，请用 *rocm* 或 *lowvram* 配置")
        elif not torch_info["cuda_ok"]:
            print("\n[FAIL] PyTorch 未识别到 GPU")
            if is_amd or platform.system() == "Linux":
                print("  AMD 云服务器修复示例:")
                print("    pip uninstall torch torchvision torchaudio -y")
                print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2")
                print("    export HSA_OVERRIDE_GFX_VERSION=10.3.0   # 部分 MI 系列需要")
            else:
                print("  NVIDIA 修复示例:")
                print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")
    else:
        print("\n[FAIL] 未安装 PyTorch: pip install -r training/requirements.txt")

    ok = torch_info.get("cuda_ok", False)
    print("\n" + "=" * 55)
    print("结论:", "可以开始 LoRA 训练" if ok else "需先配置 GPU 驱动与 ROCm/CUDA 版 PyTorch")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
