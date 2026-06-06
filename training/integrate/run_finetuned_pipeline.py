#!/usr/bin/env python3
"""使用微调后的 LoRA 模型运行完整海报 pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from poster_agent.config import Config
from poster_agent.pipeline import PosterPipeline
from training.integrate.local_llm_client import LocalLoRAClient


def main() -> int:
    parser = argparse.ArgumentParser(description="微调模型推理")
    parser.add_argument("pdf", type=Path, help="输入 PDF")
    parser.add_argument("--refiner-adapter", type=Path, required=True)
    parser.add_argument("--visual-adapter", type=Path, default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--output-name", default="finetuned_poster")
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"PDF 不存在: {args.pdf}")
        return 1

    config = Config.load()
    if not config.llm.api_key:
        print("[错误] 缺少 DeepSeek API Key。")
        print("  Parser / Visual / Commenter 等 Agent 仍需要 API。")
        print("  请在项目根目录创建 api_key.txt，或: export DEEPSEEK_API_KEY=sk-...")
        return 1

    pipeline = PosterPipeline(config)

    refiner_llm = LocalLoRAClient(args.base_model, str(args.refiner_adapter))
    pipeline.refiner.llm = refiner_llm
    if args.visual_adapter:
        visual_llm = LocalLoRAClient(args.base_model, str(args.visual_adapter))
        pipeline.visual.llm = visual_llm

    print("Refiner: 本地 LoRA")
    print(f"Visual: {'本地 LoRA' if args.visual_adapter else 'DeepSeek API'}")
    print("Parser / Semantic / Commenter: DeepSeek API")

    pipeline.run(pdf_path=args.pdf, output_name=args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
