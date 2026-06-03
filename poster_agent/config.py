"""项目配置：DeepSeek API 与海报参数."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SAMPLES_DIR = PROJECT_ROOT / "samples"


@dataclass
class LLMConfig:
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-v4-flash"
    api_key: str = ""
    temperature: float = 0.3
    max_tokens: int = 4096


@dataclass
class PosterConfig:
    width: int = 2400
    height: int = 3600
    margin: int = 60
    score_threshold: float = 0.85
    max_iterations: int = 5
    section_weights: dict[str, float] = field(
        default_factory=lambda: {
            "title": 0.12,
            "abstract": 0.10,
            "introduction": 0.12,
            "method": 0.22,
            "experiment": 0.22,
            "conclusion": 0.17,
            "reference": 0.05,
        }
    )


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    poster: PosterConfig = field(default_factory=PosterConfig)
    project_root: Path = PROJECT_ROOT
    output_dir: Path = OUTPUT_DIR

    @classmethod
    def load(cls, api_key_path: Path | None = None) -> Config:
        cfg = cls()
        key_file = api_key_path or PROJECT_ROOT / "api_key.txt"
        if key_file.exists():
            cfg.llm.api_key = key_file.read_text(encoding="utf-8").strip()
        cfg.llm.api_key = cfg.llm.api_key or os.getenv("DEEPSEEK_API_KEY", "")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        return cfg
