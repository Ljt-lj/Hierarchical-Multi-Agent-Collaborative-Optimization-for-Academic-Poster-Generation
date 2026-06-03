#!/usr/bin/env python3
"""生成用于批量测试的示例 PDF 论文."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

SAMPLES = [
    {
        "filename": "sample_transformer_nlp.pdf",
        "title": "Attention Is All You Need: Revisiting Transformer Efficiency",
        "sections": {
            "Abstract": "We propose an efficient Transformer variant achieving 92.3% accuracy on GLUE benchmark with 30% fewer parameters.",
            "Introduction": "Self-attention models dominate NLP but suffer from quadratic complexity.",
            "Method": "Our approach uses sparse attention patterns and linear projections across layers.",
            "Experiment": "Results: GLUE 92.3%, SQuAD F1 88.7%, inference speedup 2.1x over baseline.",
            "Conclusion": "Efficient attention enables scalable deployment on edge devices.",
        },
    },
    {
        "filename": "sample_diffusion_vision.pdf",
        "title": "Latent Diffusion for High-Resolution Image Synthesis",
        "sections": {
            "Abstract": "We present a latent diffusion model generating 1024px images with FID score 4.2.",
            "Introduction": "Diffusion models produce high-quality images but require massive compute.",
            "Method": "Operating in latent space reduces compute by 85% while preserving quality.",
            "Experiment": "FID 4.2 on COCO, IS 245.6, user study preference 78% over baseline.",
            "Conclusion": "Latent diffusion bridges quality and efficiency for image generation.",
        },
    },
    {
        "filename": "sample_rl_robotics.pdf",
        "title": "Multi-Agent Reinforcement Learning for Robotic Coordination",
        "sections": {
            "Abstract": "A multi-agent RL framework achieves 95% task success in warehouse robot coordination.",
            "Introduction": "Coordinating multiple robots requires decentralized decision making.",
            "Method": "We use centralized training with decentralized execution and shared value functions.",
            "Experiment": "Success rate 95%, collision rate reduced by 60%, training time 40% shorter.",
            "Conclusion": "Our method scales to 50+ agents in simulation and real-world tests.",
        },
    },
]


def build_pdf(path: Path, title: str, sections: dict[str, str]) -> None:
    doc = SimpleDocTemplate(str(path), pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(title, styles["Title"]), Spacer(1, 16)]
    for heading, body in sections.items():
        story.append(Paragraph(f"<b>{heading}</b>", styles["Heading2"]))
        story.append(Paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 10))
    doc.build(story)


def main() -> int:
    out_dir = ROOT / "samples" / "papers"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sample in SAMPLES:
        path = out_dir / sample["filename"]
        build_pdf(path, sample["title"], sample["sections"])
        print(f"Created: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
