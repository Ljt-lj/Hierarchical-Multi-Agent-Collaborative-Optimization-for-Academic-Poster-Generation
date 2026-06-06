# 学术海报 Agent 微调项目

在现有 **层级多智能体海报系统** 基础上，使用公开数据集对 **Refiner / Visual** 等 Agent 进行 LoRA 监督微调，提升内容压缩质量与插图规划能力。

## 推荐数据集（Hugging Face 现成）

| 数据集 | HF ID | 规模 | 用途 |
|--------|-------|------|------|
| **P2PInstruct** | [ASC8384/P2PInstruct](https://huggingface.co/datasets/ASC8384/P2PInstruct) | 30,460 条 | 主训练集：paper→poster 全流程 instruction-response |
| **P2PEval** | [ASC8384/P2PEval](https://huggingface.co/datasets/ASC8384/P2PEval) | 877 条 | 微调后评测（paper-poster 配对） |
| **PosterSum** | [rohitsaxena/PosterSum](https://huggingface.co/datasets/rohitsaxena/PosterSum) | 16,305 对 | Refiner 数据增强（摘要→内容树） |

论文参考：
- P2P: [2505.17104](https://arxiv.org/abs/2505.17104)
- PosterSum: [2502.17540](https://arxiv.org/abs/2502.17540)
- Paper2Poster: [2505.21497](https://arxiv.org/abs/2505.21497)

## 目录结构

```
training/
├── configs/           # 训练超参 YAML
├── data/
│   ├── download.py    # 下载 HF 数据集
│   ├── build_sft.py   # 转为 Agent 专用 JSONL
│   └── registry.json  # 数据集元信息
├── train/
│   └── train_lora.py  # LoRA 训练入口
├── eval/
│   └── evaluate.py    # 验证集 JSON 合法率等
├── integrate/
│   ├── local_llm_client.py      # 本地 LoRA 推理客户端
│   └── run_finetuned_pipeline.py # 接入完整 pipeline
└── scripts/
    └── run_all.ps1    # Windows 一键脚本
```

## 快速开始

### 1. 安装依赖

```powershell
cd C:\Users\LJTsc\Desktop\计算机图形学2.0
pip install -r training/requirements.txt
```

需要 **NVIDIA GPU + CUDA**（推荐 16GB+ 显存跑 Qwen2.5-7B LoRA）。低显存可改 `configs/*.yaml` 中 `base_model` 为 `Qwen/Qwen2.5-1.5B-Instruct`。

### 2. 下载并构建训练数据

```powershell
# 全量下载（约 2.8GB P2PInstruct）
python training/data/download.py

# 调试：只下 500 条
python training/data/download.py --max-rows 500

python training/data/build_sft.py
```

输出：
- `training/data/processed/refiner_train.jsonl`
- `training/data/processed/visual_train.jsonl`
- 对应 `*_val.jsonl`

### 3. 训练

```powershell
# Refiner Agent（内容树压缩）
python training/train/train_lora.py --config training/configs/train_refiner.yaml

# Visual Agent（插图规划）
python training/train/train_lora.py --config training/configs/train_visual.yaml

# 调试小样本
python training/train/train_lora.py --config training/configs/train_refiner.yaml --max-samples 200
```

适配器保存至 `training/outputs/refiner_lora/final/`。

### 4. 评估

```powershell
python training/eval/evaluate.py --adapter training/outputs/refiner_lora/final
```

### 5. 接入海报 pipeline

```powershell
python training/integrate/run_finetuned_pipeline.py `
  "samples/papers/real/Attention_Is_All_You_Need/1706.03762.pdf" `
  --refiner-adapter training/outputs/refiner_lora/final `
  --visual-adapter training/outputs/visual_lora/final `
  --output-name finetuned_transformer
```

## 一键脚本（Windows）

```powershell
.\training\scripts\run_all.ps1 -MaxRows 500 -Task refiner
```

## 微调任务与 Agent 映射

| Task | 对应 Agent | 数据来源 |
|------|-----------|----------|
| `refiner` | RefinerAgent | P2PInstruct 章节生成 + PosterSum 摘要增强 |
| `visual` | VisualAgent | P2PInstruct 图表/视觉描述 |
| `commenter` | CommenterAgent | P2PInstruct 评测类样本（可选） |

## 配置说明

编辑 `training/configs/train_refiner.yaml`：

- `base_model`: 基座模型（默认 Qwen2.5-7B-Instruct）
- `lora_r / lora_alpha`: LoRA 秩与缩放
- `max_seq_length`: 序列长度（Refiner 建议 4096）
- `load_in_4bit`: Linux 下 4bit 量化省显存；Windows 自动降级为 fp16

## 与 DeepSeek API 的关系

- **推理阶段**：仍可用 `api_key.txt` + DeepSeek（当前默认）
- **微调阶段**：在本地 GPU 上训练 LoRA，不消耗 DeepSeek 额度
- **部署**：`LocalLoRAClient` 替换 `LLMClient` 注入 `RefinerAgent` / `VisualAgent`

## 进阶：LLaMA-Factory（可选）

若更熟悉 [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)，可将 `data/processed/*.jsonl` 转为 sharegpt 格式后导入，数据集字段已是 `messages` 结构，迁移成本低。

## 常见问题

**Q: 下载 HF 数据集失败？**  
设置镜像：`$env:HF_ENDPOINT="https://hf-mirror.com"` 后重试。

**Q: 显存不足？**  
改用 1.5B 模型、减小 `max_seq_length`、增大 `gradient_accumulation_steps`。

**Q: 微调后效果不明显？**  
先只微调 Refiner，在 P2PEval 上对比 JSON 完整率与人工评分；再微调 Visual。
