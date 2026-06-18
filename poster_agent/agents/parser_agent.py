"""解析智能体：PDF → 原始文档树（复杂论文：长文本 + 原图筛选）."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import fitz

from poster_agent.agents.figure_curator import (
    assign_figures_to_raw_tree,
    attach_caption_hints,
    build_figure_catalog,
    catalog_and_filter_images,
    extract_figure_captions,
)
from poster_agent.extract.figure_page_renderer import (
    merge_poster_figure_assets,
    render_figure_composites,
    render_figure_focus_panels,
)
from poster_agent.extract.table_extractor import extract_key_tables
from poster_agent.config import PosterConfig
from poster_agent.llm_client import LLMClient
from poster_agent.render.language import detect_language, language_instruction
from poster_agent.models.trees import RawNode

logger = logging.getLogger(__name__)

# 单次 LLM 结构化调用字符上限（过长会导致空响应或 JSON 失败）
LLM_STRUCTURE_CHARS = 22_000
SECTION_PREVIEW_CHARS = 900

TOP_SECTIONS = (
    "abstract",
    "introduction",
    "background",
    "related work",
    "methods",
    "method",
    "materials and methods",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "references",
    "acknowledgements",
    "摘要",
    "引言",
    "背景",
    "方法",
    "结果",
    "讨论",
    "结论",
    "参考文献",
)


class ParserAgent:
    def __init__(self, llm: LLMClient, poster_config: PosterConfig | None = None):
        self.llm = llm
        self.config = poster_config or PosterConfig()

    def parse(self, pdf_path: Path) -> RawNode:
        text_blocks, image_paths = self._extract_pdf(pdf_path)
        lang = detect_language(text_blocks)
        structured = self._structure_document(text_blocks, lang)
        structured.authors = extract_authors(text_blocks)
        self._enrich_with_figures(structured, text_blocks, image_paths, lang, pdf_path)
        return structured

    def _structure_document(self, text: str, lang: str) -> RawNode:
        if len(text) > LLM_STRUCTURE_CHARS:
            logger.info("Long paper (%d chars), using chunked parser", len(text))
            try:
                return self._structure_chunked(text, lang)
            except Exception as exc:
                logger.warning("Chunked parser failed: %s, using heuristic fallback", exc)
                return _build_heuristic_tree(text)

        try:
            return self._structure_with_llm(text, lang)
        except Exception as exc:
            logger.warning("Single-shot parser failed: %s, trying chunked/heuristic", exc)
            try:
                return self._structure_chunked(text, lang)
            except Exception:
                return _build_heuristic_tree(text)

    def _structure_with_llm(self, text: str, lang: str = "en") -> RawNode:
        llm_text = text if len(text) <= LLM_STRUCTURE_CHARS else _smart_truncate(text, LLM_STRUCTURE_CHARS)
        system = (
            "You are an academic paper parsing expert. Build a hierarchical document tree. "
            "Keep Results/Methods subsections as children when present. "
            "Preserve key paragraphs in content (not one-sentence summaries). "
            "Output JSON: {title, content, level, children:[...]}. "
            + language_instruction(lang)
        )
        if lang == "zh":
            system = (
                "构建学术论文层级文档树，保留 Results/Methods 子节。"
                "content 保留关键段落。输出 JSON：{title, content, level, children:[...]}。"
                + language_instruction(lang)
            )
        data = self.llm.chat_json(
            system,
            f"Paper text:\n{llm_text}" if lang == "en" else f"论文文本：\n{llm_text}",
            max_tokens=8192,
        )
        if isinstance(data, list) and data:
            data = data[0]
        return RawNode.from_dict(data)

    def _structure_chunked(self, text: str, lang: str) -> RawNode:
        """长论文：启发式分节 + 各节预览送 LLM 构建树（避免整篇超长 JSON）."""
        sections = split_paper_sections(text)
        title = guess_paper_title(text, sections)
        preview_blocks: list[str] = []
        for name, body in sections.items():
            if name == "__title__":
                continue
            preview_blocks.append(f"## {name}\n{(body or '')[:SECTION_PREVIEW_CHARS]}")
        preview = "\n\n".join(preview_blocks)

        system = (
            "Build a hierarchical document tree from section previews of a long paper. "
            "Rules: 1) Use provided section titles; 2) For Results/Methods with multiple "
            "logical parts in preview, create children subsections; "
            "3) Put full preview text in each node content field; "
            "4) level: 0 for root, 1 for top sections, 2 for subsections. "
            "Output JSON: {title, content, level, children:[...]}. "
            + language_instruction(lang)
        )
        if lang == "zh":
            system = (
                "根据长论文各节预览构建层级文档树；Results/Methods 可含 children 子节。"
                "每节点 content 填预览全文。输出 JSON：{title, content, level, children}。"
                + language_instruction(lang)
            )
        user = f"Paper title: {title}\n\nSection previews:\n{preview[:18000]}"
        data = self.llm.chat_json(system, user, max_tokens=8192, temperature=0.2)
        if isinstance(data, list) and data:
            data = data[0]

        tree = RawNode.from_dict(data)
        if not tree.title:
            tree.title = title
        _merge_full_section_text(tree, sections)
        if not tree.children:
            tree.children = [
                RawNode(title=k, content=v, level=1)
                for k, v in sections.items()
                if k != "__title__" and v.strip()
            ]
        return tree

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
                    if pix.width < 100 or pix.height < 100:
                        continue
                    out = image_dir / f"page{page_idx + 1}_img{img_idx + 1}.png"
                    pix.save(out)
                    image_paths.append(str(out))
                except Exception:
                    continue

        doc.close()
        full_text = "\n".join(parts)
        limit = self.config.parser_max_chars
        if len(full_text) > limit:
            full_text = _smart_truncate(full_text, limit)
        return full_text, image_paths

    def _enrich_with_figures(
        self,
        node: RawNode,
        full_text: str,
        image_paths: list[str],
        lang: str,
        pdf_path: Path,
    ) -> None:
        image_dir = pdf_path.parent / f"{pdf_path.stem}_images"
        captions = extract_figure_captions(full_text)
        doc = fitz.open(pdf_path)
        from poster_agent.extract.figure_page_renderer import extract_figure_captions_from_pdf

        for num, cap in extract_figure_captions_from_pdf(doc).items():
            if num not in captions or len(cap) > len(captions.get(num, "")):
                captions[num] = cap
        doc.close()
        embedded = catalog_and_filter_images(image_paths)
        attach_caption_hints(embedded, captions)
        composites = render_figure_composites(pdf_path, image_dir, captions)
        focus = render_figure_focus_panels(pdf_path, image_dir, captions)
        assets = merge_poster_figure_assets(focus, composites, embedded)
        if focus:
            logger.info("Rendered %d focus panel(s) for poster", len(focus))
        if composites:
            logger.info("Rendered %d composite figure(s) from PDF pages", len(composites))
        assign_figures_to_raw_tree(
            node,
            assets,
            self.llm,
            lang,
            max_per_section=self.config.max_figures_per_section,
        )
        node.figure_catalog = build_figure_catalog(assets)
        node.tables = [t.to_dict() for t in extract_key_tables(full_text, llm=self.llm, language=lang)]

    def parse_text(self, title: str, sections: dict[str, str]) -> RawNode:
        children = [RawNode(title=k, content=v, level=2) for k, v in sections.items()]
        return RawNode(title=title, content=sections.get("Abstract", ""), level=1, children=children)


def split_paper_sections(text: str) -> dict[str, str]:
    """按常见论文章节标题切分正文."""
    lines = text.splitlines()
    header_pat = re.compile(
        r"^(" + "|".join(re.escape(h) for h in TOP_SECTIONS if h not in ("method", "methods", "方法")) + r")\.?\s*$",
        re.IGNORECASE,
    )
    method_pat = re.compile(r"^(Methods|Method|Materials and methods|方法)\.?\s*$", re.IGNORECASE)
    hits: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        if re.search(r"nature\s+methods", stripped, re.I):
            continue
        m = header_pat.match(stripped) or method_pat.match(stripped)
        if m:
            hits.append((i, _normalize_section_name(m.group(1))))

    sections: dict[str, str] = {}
    if hits:
        pre = "\n".join(lines[: hits[0][0]]).strip()
        _assign_preamble(sections, pre)
        for idx, (line_no, name) in enumerate(hits):
            end = hits[idx + 1][0] if idx + 1 < len(hits) else len(lines)
            body = "\n".join(lines[line_no + 1 : end]).strip()
            if name in sections:
                sections[name] = sections[name] + "\n\n" + body
            else:
                sections[name] = body
    else:
        sections["Body"] = text.strip()
    return sections


def _assign_preamble(sections: dict[str, str], pre: str) -> None:
    if len(pre) < 200:
        return
    intro_markers = [
        "Protein–protein interactions",
        "Protein-protein interactions",
        "Several tools were developed",
        "In this paper, we introduce",
        "本文提出",
        "引言",
    ]
    for marker in intro_markers:
        pos = pre.find(marker)
        if pos > 100:
            sections["Abstract"] = pre[:pos].strip()
            sections["Introduction"] = pre[pos:].strip()
            return
    sections["Abstract"] = pre[: min(3500, len(pre))]
    if len(pre) > 3500:
        sections["Introduction"] = pre[3500:].strip()


def _normalize_section_name(name: str) -> str:
    n = name.strip().lower()
    mapping = {
        "abstract": "Abstract",
        "introduction": "Introduction",
        "background": "Introduction",
        "related work": "Introduction",
        "methods": "Methods",
        "method": "Methods",
        "materials and methods": "Methods",
        "results": "Results",
        "discussion": "Discussion",
        "conclusion": "Conclusion",
        "conclusions": "Conclusion",
        "references": "References",
        "acknowledgements": "References",
        "摘要": "Abstract",
        "引言": "Introduction",
        "背景": "Introduction",
        "方法": "Methods",
        "结果": "Results",
        "讨论": "Discussion",
        "结论": "Conclusion",
        "参考文献": "References",
    }
    return mapping.get(n, name.strip().title())


def guess_paper_title(text: str, sections: dict[str, str]) -> str:
    if title := sections.get("__title__"):
        return title
    for line in text.splitlines()[:40]:
        line = line.strip()
        if not line or len(line) < 12:
            continue
        if line.lower() in TOP_SECTIONS:
            continue
        if re.search(r"nature methods|doi\.org|https?://", line, re.I):
            continue
        if re.match(r"^\d+$", line):
            continue
        if len(line.split()) >= 4:
            return line[:200]
    return "Untitled Paper"


def extract_authors(text: str) -> str:
    """从论文首页提取作者行（Nature/AAAI 常见格式）."""
    lines = [ln.strip() for ln in text.splitlines()[:80] if ln.strip()]
    author_lines: list[str] = []
    for i, line in enumerate(lines[:35]):
        if re.search(r"nature methods|doi\.org|https?://|volume \d+|article\s*$", line, re.I):
            continue
        if len(line) > 180 or len(line.split()) > 25:
            continue
        if re.match(r"^(abstract|introduction|keywords|correspondence)\b", line, re.I):
            break
        # 作者行：含逗号分隔姓名，或 "and" 连接，且不像标题
        if re.search(r",\s*[A-Z]| and [A-Z]|\d+\s*,\s*\d+", line):
            if not re.search(r"university|institute|department|laboratory|@|fig\.|table", line, re.I):
                author_lines.append(line)
                if len(author_lines) >= 2:
                    break
        elif author_lines and len(line.split()) <= 8 and not line.endswith("."):
            author_lines.append(line)
            break
    if not author_lines:
        for line in lines[1:12]:
            if 20 < len(line) < 120 and "," in line and line[0].isupper():
                if not re.search(r"integrat|predict|method|structure|protein", line, re.I):
                    return line[:160]
    return " · ".join(author_lines)[:220] if author_lines else ""


def _build_heuristic_tree(text: str) -> RawNode:
    sections = split_paper_sections(text)
    title = guess_paper_title(text, sections)
    abstract = sections.get("Abstract", "")
    children = [
        RawNode(title=k, content=v, level=1)
        for k, v in sections.items()
        if k not in ("Abstract", "__title__") and v.strip()
    ]
    return RawNode(title=title, content=abstract, level=0, children=children)


def _merge_full_section_text(node: RawNode, sections: dict[str, str]) -> None:
    """用启发式切分的完整正文覆盖 LLM 预览片段."""

    def match_section(title: str) -> str | None:
        t = title.lower()
        for key, body in sections.items():
            if key == "__title__":
                continue
            if key.lower() in t or t in key.lower():
                return body
        return None

    def walk(n: RawNode) -> None:
        body = match_section(n.title)
        if body and len(body) > len(n.content or "") + 50:
            n.content = body
        for c in n.children:
            walk(c)

    walk(node)
    if not node.content:
        node.content = sections.get("Abstract", "")


def _smart_truncate(text: str, max_chars: int) -> str:
    """保留首尾与 Results/Method 关键页，而非简单截断开头."""
    if len(text) <= max_chars:
        return text
    pages = re.split(r"\n--- Page (\d+) ---\n", text)
    if len(pages) < 3:
        return text[:max_chars] + "\n...[truncated]..."

    chunks: list[tuple[int, str]] = []
    i = 1
    while i < len(pages) - 1:
        try:
            num = int(pages[i])
            body = pages[i + 1]
            chunks.append((num, body))
            i += 2
        except ValueError:
            i += 1

    priority: list[tuple[int, str]] = []
    tail: list[tuple[int, str]] = []
    for num, body in chunks:
        low = body.lower()
        score = 0
        if any(k in low for k in ("abstract", "introduction", "摘要", "引言")):
            score += 3
        if any(k in low for k in ("method", "approach", "framework", "方法", "模型")):
            score += 4
        if any(k in low for k in ("result", "experiment", "evaluation", "结果", "实验")):
            score += 5
        if re.search(r"fig\.?\s*\d", low):
            score += 2
        if score >= 4:
            priority.append((num, body))
        elif num <= 4 or num >= max(c[0] for c in chunks) - 3:
            tail.append((num, body))

    selected = priority + tail
    seen: set[int] = set()
    ordered: list[str] = []
    total = 0
    for num, body in sorted(selected, key=lambda x: x[0]):
        if num in seen:
            continue
        seen.add(num)
        block = f"\n--- Page {num} ---\n{body}"
        if total + len(block) > max_chars:
            remain = max_chars - total - 40
            if remain > 500:
                ordered.append(block[:remain] + "\n...[page truncated]...")
            break
        ordered.append(block)
        total += len(block)
    return "".join(ordered) + "\n...[document truncated]..."
