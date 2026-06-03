"""解析智能体：PDF → 原始文档树."""

from __future__ import annotations

import re
from pathlib import Path

import fitz

from poster_agent.llm_client import LLMClient
from poster_agent.render.language import detect_language, language_instruction
from poster_agent.models.trees import RawNode


class ParserAgent:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def parse(self, pdf_path: Path) -> RawNode:
        text_blocks, image_paths = self._extract_pdf(pdf_path)
        lang = detect_language(text_blocks)
        structured = self._structure_with_llm(text_blocks, lang)
        self._attach_images(structured, image_paths)
        return structured

    def _extract_pdf(self, pdf_path: Path) -> tuple[str, list[str]]:
        doc = fitz.open(pdf_path)
        parts: list[str] = []
        image_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        image_dir.mkdir(exist_ok=True)
        image_paths: list[str] = []

        for page_idx, page in enumerate(doc):
            parts.append(f"\n--- Page {page_idx + 1} ---\n")
            parts.append(page.get_text("text"))
            for img_idx, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n >= 5:
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 80 or pix.height < 80:
                        continue
                    out = image_dir / f"page{page_idx + 1}_img{img_idx + 1}.png"
                    pix.save(out)
                    image_paths.append(str(out))
                except Exception:
                    continue

        doc.close()
        full_text = "\n".join(parts)
        if len(full_text) > 12000:
            full_text = full_text[:12000] + "\n...[truncated]..."
        return full_text, image_paths

    def _structure_with_llm(self, text: str, lang: str = "en") -> RawNode:
        system = (
            "You are an academic paper parsing expert. Structure the paper text into a hierarchical document tree. "
            "Output JSON: {title, content, level, children:[...]}. "
            "Organize children by sections (Abstract, Introduction, Method, Experiments, Conclusion, etc.). "
            "Keep key original text in each node content field. "
            + language_instruction(lang)
        )
        if lang == "zh":
            system = (
                "你是学术论文解析专家。将论文文本结构化为层级文档树。"
                "输出 JSON 格式：{title, content, level, children:[...]}。"
                "children 按论文章节组织。每个节点 content 保留关键段落原文。"
                + language_instruction(lang)
            )
        data = self.llm.chat_json(
            system,
            f"Paper text:\n{text}" if lang == "en" else f"论文文本：\n{text}",
        )
        if isinstance(data, list) and data:
            data = data[0]
        return RawNode.from_dict(data)

    def _attach_images(self, node: RawNode, images: list[str]) -> None:
        if not images:
            return
        exp_keywords = re.compile(r"experiment|result|figure|实验|结果", re.I)
        method_keywords = re.compile(r"method|approach|方法", re.I)

        def walk(n: RawNode) -> None:
            title = n.title or ""
            if exp_keywords.search(title) and not n.images:
                n.images.extend(images[:3])
            elif method_keywords.search(title) and len(images) > 1:
                n.images.extend(images[:2])
            elif re.search(r"intro|introduction|引言", title, re.I) and len(images) > 3:
                n.images.append(images[0])
            for c in n.children:
                walk(c)

        walk(node)
        if not any(n.images for n in _flatten_raw(node)):
            node.images = images[:1]

    def parse_text(self, title: str, sections: dict[str, str]) -> RawNode:
        """无 PDF 时的演示解析."""
        children = [
            RawNode(title=k, content=v, level=2) for k, v in sections.items()
        ]
        return RawNode(title=title, content=sections.get("Abstract", ""), level=1, children=children)


def _flatten_raw(node: RawNode) -> list[RawNode]:
    out = [node]
    for c in node.children:
        out.extend(_flatten_raw(c))
    return out
