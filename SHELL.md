# 命令手册：海报生成 & 模型微调

> 项目根目录：`C:\Users\LJTsc\Desktop\计算机图形学2.0`  
> 所有命令在 **PowerShell** 中执行，先 `cd` 到项目根目录。

---

## 是否需要 GPU？

| 任务 | 是否需要 GPU | 说明 |
|------|-------------|------|
| **海报生成（现有 pipeline）** | 否 | 调用 DeepSeek API，只需网络和 `api_key.txt` |
| **LoRA 微调训练** | **是（强烈建议）** | 7B 模型全量 LoRA 在 CPU 上几乎不可行 |
| **微调模型本地推理** | **是** | 加载 Qwen + LoRA 需 GPU（至少 6GB 显存） |
| **下载 / 构建数据集** | 否 | 仅需磁盘与网络 |

**结论：微调需要 GPU。支持 NVIDIA（CUDA）或 AMD 云端（ROCm），不支持纯 CPU 训练。**

### 显卡类型对照

| 环境 | 驱动/工具 | PyTorch 安装 | 本项目配置 |
|------|-----------|-------------|-----------|
| **NVIDIA** | `nvidia-smi` | `cu124` 轮子 | `train_refiner.yaml` |
| **AMD 云** | `rocm-smi` / `amd-smi` | `rocm6.2` 轮子 | `train_refiner_rocm.yaml` |
| **本机 Windows 无 GPU** | — | — | 海报用 API；训练在云端 AMD/NVIDIA |

### 显存与推荐配置

| 显存 | NVIDIA 配置 | AMD 云配置 |
|------|------------|-----------|
| ≥ 24 GB | `train_refiner.yaml` (7B) | `train_refiner_rocm.yaml` (7B) |
| 12~16 GB | `train_refiner.yaml` 或 lowvram | `train_refiner_rocm.yaml`，OOM 则 lowvram |
| 6~8 GB | `train_refiner_lowvram.yaml` (1.5B) | `train_refiner_lowvram.yaml` (1.5B) |

**AMD 特别注意：不要使用 4bit 量化（bitsandbytes 仅 NVIDIA）；必须用 `*_rocm.yaml` 或 `*_lowvram.yaml`。**

### 本机 Windows（开发机）检测结果

```
PyTorch: 2.9.1+cu126
GPU 可用: False
```

说明：**本机不负责训练**，在云端 AMD 实例上跑训练；本机继续用 DeepSeek API 生成海报即可。

---

---

## 0-A. AMD 云端 ROCm 环境（推荐给你）

> 在 **云端 Linux SSH** 里执行（不是 Windows 本机）。  
> 镜像建议：Ubuntu 22.04 + ROCm 6.x 预装，或官方 PyTorch ROCm Docker。

### A.1 登录云服务器后检查

```bash
cd ~/计算机图形学2.0          # 或 git clone 后的项目路径
bash training/scripts/setup_rocm.sh
python training/scripts/check_gpu.py
```

应看到 `GPU 可用 (torch.cuda): True` 且设备名含 AMD / gfx。

### A.2 安装 ROCm 版 PyTorch（若 GPU 不可用）

```bash
pip uninstall torch torchvision torchaudio bitsandbytes -y

# ROCm 6.2（按云厂商文档选 rocm6.1 / rocm6.2）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# 部分 MI 系列实例需要（按云厂商说明调整）
export HSA_OVERRIDE_GFX_VERSION=10.3.0

python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### A.3 AMD 云完整训练流程

```bash
export HF_ENDPOINT=https://hf-mirror.com

pip install -r requirements.txt
pip install -r training/requirements.txt
pip uninstall bitsandbytes -y    # AMD 必须去掉

# 1) 数据
python training/data/download.py --max-rows 500    # 试跑；全量去掉 --max-rows
python training/data/build_sft.py

# 2) 训练 Refiner（AMD 专用配置，fp16 LoRA，无 4bit）
python training/train/train_lora.py --config training/configs/train_refiner_rocm.yaml

# 3) 训练 Visual（可选）
python training/train/train_lora.py --config training/configs/train_visual_rocm.yaml

# 4) 评估
python training/eval/evaluate.py \
  --adapter training/outputs/refiner_lora_rocm/final \
  --base-model Qwen/Qwen2.5-7B-Instruct

# 5) 用微调模型生成海报（PDF 需上传到云或挂载数据盘）
python training/integrate/run_finetuned_pipeline.py \
  /path/to/paper.pdf \
  --refiner-adapter training/outputs/refiner_lora_rocm/final \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-name amd_finetuned_poster
```

### A.4 训练完成后拉回本机

```bash
# 在云服务器打包
tar czf refiner_lora_rocm.tar.gz training/outputs/refiner_lora_rocm/final

# 本机 Windows（PowerShell + scp 示例）
scp user@云IP:~/计算机图形学2.0/refiner_lora_rocm.tar.gz .
```

### A.5 AMD 常见问题

| 现象 | 处理 |
|------|------|
| `bitsandbytes` 报错 | `pip uninstall bitsandbytes -y`，用 `*_rocm.yaml` |
| OOM 显存不足 | 改用 `train_refiner_lowvram.yaml` 或减小 `max_seq_length` |
| `torch.cuda False` | 重装 ROCm 版 PyTorch；问云厂商要 ROCm 版本号 |
| gfx 不匹配 | 设置 `HSA_OVERRIDE_GFX_VERSION`（见云文档） |

---

## 0-B. NVIDIA 环境准备

### B.1 检查 GPU

```powershell
cd C:\Users\LJTsc\Desktop\计算机图形学2.0
python training/scripts/check_gpu.py
```

### B.2 安装 NVIDIA 驱动

1. 打开 https://www.nvidia.com/Download/index.aspx 下载并安装驱动  
2. 重启后验证：

```powershell
nvidia-smi
```

### B.3 安装 CUDA 版 PyTorch

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### B.4 安装项目依赖

```powershell
# 海报生成（主项目）
pip install -r requirements.txt

# 微调训练（额外）
pip install -r training/requirements.txt
```

### B.5 Hugging Face 镜像

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

---

## 1. 海报生成（无需 GPU）

### 1.1 指定 arXiv 论文完整跑一遍

```powershell
cd C:\Users\LJTsc\Desktop\计算机图形学2.0

# 首次：自动下载 PDF
python experiments/real_paper_test.py --arxiv-id 1706.03762 --count 1

# 已下载过 PDF
python experiments/real_paper_test.py --skip-download --arxiv-id 1706.03762 --count 1
```

### 1.2 本地 PDF 单篇生成

```powershell
python main.py "路径\到\论文.pdf" --output-name My_Paper
```

### 1.3 从已有 content JSON 快速重渲染

```powershell
python tools/rerender_poster.py `
  "outputs\real_paper_test\...\*_content_iter5.json" `
  --raw-json "outputs\real_paper_test\...\*_raw_tree.json" `
  --output-name "result" `
  --prefix "rerender"
```

---

## 2. 微调全流程（需要 GPU）

按顺序执行以下命令。

### 2.1 下载公开数据集

```powershell
cd C:\Users\LJTsc\Desktop\计算机图形学2.0
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 全量（P2PInstruct ~2.8GB + PosterSum + P2PEval）
python training/data/download.py

# 调试：只下载 500 条
python training/data/download.py --max-rows 500

# 只下载某一集
python training/data/download.py --dataset p2p_instruct
python training/data/download.py --dataset poster_sum
python training/data/download.py --dataset p2p_eval
```

数据集来源：
- [P2PInstruct](https://huggingface.co/datasets/ASC8384/P2PInstruct) — 主训练集  
- [PosterSum](https://huggingface.co/datasets/rohitsaxena/PosterSum) — Refiner 增强  
- [P2PEval](https://huggingface.co/datasets/ASC8384/P2PEval) — 评测  

### 2.2 构建 SFT 训练集

```powershell
python training/data/build_sft.py
```

输出目录：`training/data/processed/`
- `refiner_train.jsonl` / `refiner_val.jsonl`
- `visual_train.jsonl` / `visual_val.jsonl`

### 2.3 LoRA 训练 — Refiner Agent

```powershell
# 16GB+ 显存（Qwen2.5-7B）
python training/train/train_lora.py --config training/configs/train_refiner.yaml

# 6~8GB 显存（Qwen2.5-1.5B）
python training/train/train_lora.py --config training/configs/train_refiner_lowvram.yaml

# 快速试跑 200 条
python training/train/train_lora.py --config training/configs/train_refiner_lowvram.yaml --max-samples 200
```

权重输出：`training/outputs/refiner_lora/final/` 或 `training/outputs/refiner_lora_1.5b/final/`

### 2.4 LoRA 训练 — Visual Agent（可选）

```powershell
python training/train/train_lora.py --config training/configs/train_visual.yaml
# 或低显存
python training/train/train_lora.py --config training/configs/train_visual_lowvram.yaml
```

### 2.5 验证集评估

```powershell
python training/eval/evaluate.py `
  --adapter training/outputs/refiner_lora/final `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --max-samples 50
```

### 2.6 用微调模型跑完整海报 pipeline

```powershell
python training/integrate/run_finetuned_pipeline.py `
  "samples\papers\real\Attention_Is_All_You_Need\1706.03762.pdf" `
  --refiner-adapter training/outputs/refiner_lora/final `
  --visual-adapter training/outputs/visual_lora/final `
  --base-model Qwen/Qwen2.5-7B-Instruct `
  --output-name finetuned_transformer
```

---

## 3. 一键脚本（Windows）

```powershell
cd C:\Users\LJTsc\Desktop\计算机图形学2.0
.\training\scripts\run_all.ps1 -MaxRows 500 -Task refiner
```

参数：
- `-MaxRows 500`：调试时限制下载条数（0 = 全量）
- `-Task refiner` | `visual` | `all`

---

## 4. 云 GPU 方案

### 4.1 AMD 云（你的情况）

- 训练在 **云端 Linux + ROCm** 完成 → 见上文 **「0-A. AMD 云端 ROCm 环境」**
- 本机 Windows 只做开发、API 海报生成、下载训练好的 LoRA 权重

### 4.2 NVIDIA 云

- [AutoDL](https://www.autodl.com/) — NVIDIA RTX 4090 / A100  
- [Google Colab Pro](https://colab.research.google.com/) — T4  

### 4.3 NVIDIA 云最小流程

```bash
# Linux 示例
git clone <你的仓库> && cd 计算机图形学2.0
pip install -r requirements.txt -r training/requirements.txt

export HF_ENDPOINT=https://hf-mirror.com
python training/scripts/check_gpu.py

python training/data/download.py --max-rows 500   # 先小规模试
python training/data/build_sft.py
python training/train/train_lora.py --config training/configs/train_refiner.yaml

# 训练完成后把 training/outputs/ 下载回本地
```

---

## 5. 常用命令速查

```powershell
# AMD 云训练（SSH 里）
python training/scripts/check_gpu.py
python training/train/train_lora.py --config training/configs/train_refiner_rocm.yaml

# 本机 Windows：海报 API 生成
python experiments/real_paper_test.py --skip-download --arxiv-id 1706.03762 --count 1

# 数据：下载 → 构建
python training/data/download.py --max-rows 500
python training/data/build_sft.py

# 训练：Refiner（低显存）
python training/train/train_lora.py --config training/configs/train_refiner_lowvram.yaml --max-samples 200

# 评估
python training/eval/evaluate.py --adapter training/outputs/refiner_lora_1.5b/final --base-model Qwen/Qwen2.5-1.5B-Instruct
```

---

## 6. 故障排查

| 现象 | 处理 |
|------|------|
| `CUDA/ROCm False` | AMD：装 rocm6.2 轮子；NVIDIA：装 cu124 轮子 |
| `bitsandbytes` 报错 | AMD 必须卸载 bitsandbytes |
| 显存 OOM | 改用 `*_lowvram.yaml`，或减小 `max_seq_length` / `batch_size` |
| HF 下载失败 | `$env:HF_ENDPOINT="https://hf-mirror.com"` |
| arXiv 429 | 加 `--skip-download`，或稍后重试 |
| 训练集不存在 | 先跑 `download.py` 再跑 `build_sft.py` |

---

## 7. 相关文档

- 微调详细说明：`training/README.md`
- 实验方案：`project.md`
