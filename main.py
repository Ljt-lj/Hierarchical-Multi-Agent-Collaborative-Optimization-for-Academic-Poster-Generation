#!/usr/bin/env python3
"""主入口：运行学术海报生成管道."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from poster_agent.config import Config
from poster_agent.pipeline import PosterPipeline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="基于层级多智能体协作的学术海报生成系统"
    )
    parser.add_argument(
        "pdf",
        nargs="?",
        help="输入 PDF 论文路径（省略则使用演示模式）",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="使用内置演示论文数据",
    )
    parser.add_argument(
        "--output-name",
        default="auto",
        help="输出文件前缀（默认 auto：按时间戳+论文名自动生成）",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DeepSeek 模型名（默认 deepseek-v4-flash）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="全局控制评分阈值（默认 0.85）",
    )
    args = parser.parse_args()

    config = Config.load()
    if not config.llm.api_key:
        print("错误：未找到 API 密钥。请在 api_key.txt 或环境变量 DEEPSEEK_API_KEY 中配置。", file=sys.stderr)
        return 1

    if args.model:
        config.llm.model = args.model
    if args.threshold is not None:
        config.poster.score_threshold = args.threshold

    pdf_path = Path(args.pdf) if args.pdf else None
    demo = args.demo or pdf_path is None

    if pdf_path and not pdf_path.exists():
        print(f"错误：PDF 文件不存在: {pdf_path}", file=sys.stderr)
        return 1

    pipeline = PosterPipeline(config)
    pipeline.run(pdf_path=pdf_path, demo=demo, output_name=args.output_name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
