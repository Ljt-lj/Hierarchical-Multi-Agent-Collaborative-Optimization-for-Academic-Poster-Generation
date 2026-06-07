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
    inner_paint_passes: int = 1  # Painter–Commenter 内层重绘次数（Paper2Poster）
    min_body_font: int = 34
    min_title_font: int = 42
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
            cfg.llm.api_key = _read_api_key_file(key_file)
        cfg.llm.api_key = cfg.llm.api_key or os.getenv("DEEPSEEK_API_KEY", "")
        cfg.output_dir.mkdir(parents=True, exist_ok=True)
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        return cfg


def _read_api_key_file(path: Path) -> str:
    for enc in ("utf-8", "utf-8-sig", "gbk", "latin-1"):
        try:
            text = path.read_text(encoding=enc).strip()
            if text:
                for line in text.splitlines():
                    line = line.strip()
                    if line.startswith("sk-"):
                        return line
                return text
        except (UnicodeDecodeError, OSError):
            continue
    return ""
