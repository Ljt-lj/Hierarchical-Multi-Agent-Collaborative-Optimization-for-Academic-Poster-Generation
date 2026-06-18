"""从论文正文提取关键实验数据表（Table / Supplementary Table）."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poster_agent.llm_client import LLMClient
from poster_agent.render.language import language_instruction


@dataclass
class PaperTable:
    caption: str
    headers: list[str]
    rows: list[list[str]]
    table_num: int = 0
    source: str = "extracted"

    def to_dict(self) -> dict:
        return {
            "caption": self.caption,
            "headers": self.headers,
            "rows": self.rows,
            "table_num": self.table_num,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PaperTable:
        return cls(
            caption=str(data.get("caption", "")),
            headers=[str(h) for h in data.get("headers") or []],
            rows=[[str(c) for c in row] for row in data.get("rows") or []],
            table_num=int(data.get("table_num", 0)),
            source=str(data.get("source", "extracted")),
        )


def extract_table_captions(text: str) -> dict[int, str]:
    captions: dict[int, str] = {}
    patterns = [
        r"(?:Table|TABLE)\s*(\d+)[a-z]?\s*[|.:\-–—]?\s*([^\n]{15,200})",
        r"(?:Supplementary\s+Table|Suppl\.?\s*Table)\s*(\d+)[a-z]?\s*[|.:\-–—]?\s*([^\n]{15,180})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text, re.I):
            num = int(m.group(1))
            cap = re.sub(r"\s+", " ", m.group(2)).strip()
            if num not in captions or len(cap) > len(captions[num]):
                captions[num] = cap
    return captions


def extract_key_tables(
    text: str,
    *,
    llm: LLMClient | None = None,
    language: str = "en",
    max_tables: int = 3,
) -> list[PaperTable]:
    """启发式 + 可选 LLM 提取 1–3 张关键 benchmark 表."""
    captions = extract_table_captions(text)
    tables: list[PaperTable] = []

    # 正文中显式 Table 引用附近的数值块
    for num, cap in sorted(captions.items())[:max_tables]:
        block = _table_block_near_caption(text, num)
        parsed = _parse_tabular_block(block, cap, num)
        if parsed:
            tables.append(parsed)

    # 常见 benchmark 句群（DockQ / BLEU / success rate）
    if len(tables) < max_tables:
        bench = _extract_benchmark_table(text, language)
        if bench and not _duplicate_table(tables, bench):
            tables.append(bench)

    if len(tables) < max_tables:
        restraint = _extract_restraint_dockq_table(text, language)
        if restraint and not _duplicate_table(tables, restraint):
            tables.append(restraint)

    if llm and len(tables) < max_tables:
        llm_table = _llm_extract_table(text, llm, language)
        if llm_table and not _duplicate_table(tables, llm_table):
            tables.append(llm_table)

    return tables[:max_tables]


def _table_block_near_caption(text: str, table_num: int) -> str:
    pat = re.compile(rf"(?:Table|TABLE)\s*{table_num}[a-z]?", re.I)
    m = pat.search(text)
    if not m:
        return ""
    start = m.end()
    return text[start : start + 1200]


def _parse_tabular_block(block: str, caption: str, num: int) -> PaperTable | None:
    if not block.strip():
        return None
    lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    rows: list[list[str]] = []
    for ln in lines[:12]:
        if re.search(r"nature methods|doi\.org|https?://", ln, re.I):
            continue
        cells = re.split(r"\s{2,}|\t|\|", ln)
        cells = [c.strip() for c in cells if c.strip()]
        if len(cells) >= 3 and _row_has_number(cells):
            rows.append(cells[:6])
        if len(rows) >= 6:
            break
    if len(rows) < 2:
        return None
    headers = rows[0]
    body = rows[1:]
    if not _row_has_number(headers) and body:
        return PaperTable(caption=caption[:120], headers=headers, rows=body, table_num=num)
    return None


def _extract_benchmark_table(text: str, language: str) -> PaperTable | None:
    en = language == "en"
    methods: list[str] = []
    dockqs: list[str] = []
    for m in re.finditer(
        r"(AFM|AF3|GRASP|AlphaLink|HADDOCK|ColabDock|ClusPro)[^.;\n]{0,60}?"
        r"(?:mean\s+)?(?:DockQ|dockq)[^0-9]{0,12}(\d+\.?\d*)",
        text,
        re.I,
    ):
        name = m.group(1).upper() if m.group(1).isupper() else m.group(1)
        methods.append(name)
        dockqs.append(m.group(2))
    if len(methods) < 2:
        for m in re.finditer(
            r"(AFM|AF3|GRASP|AlphaLink|HADDOCK|ColabDock|ClusPro)\s+[^0-9\n]{0,30}(\d+\.\d{2})",
            text,
            re.I,
        ):
            methods.append(m.group(1).upper() if m.group(1).isupper() else m.group(1))
            dockqs.append(m.group(2))
    if len(methods) < 2:
        return None
    pairs = list(dict.fromkeys(zip(methods, dockqs)))[:6]
    return PaperTable(
        caption="Benchmark DockQ comparison" if en else "基准 DockQ 对比",
        headers=["Method" if en else "方法", "DockQ"],
        rows=[[a, b] for a, b in pairs],
        table_num=0,
        source="benchmark_heuristic",
    )


def _extract_restraint_dockq_table(text: str, language: str) -> PaperTable | None:
    """从正文提取 RPR/IR 数量与 DockQ（对应论文 benchmark 数据）."""
    en = language == "en"
    rows: list[list[str]] = []
    for m in re.finditer(
        r"(\d+)\s+(?:contact RPRs|IRs)[^.;\n]{0,120}?(?:mean\s+)?DockQ[^0-9]{0,12}(\d+\.?\d*)",
        text,
        re.I,
    ):
        rows.append([f"{m.group(1)} restraints", m.group(2)])
    m2 = re.search(
        r"two inter-chain contact RPRs[^.;\n]{0,100}?DockQ[^0-9]{0,12}(\d+\.?\d*)",
        text,
        re.I,
    )
    if m2:
        rows.insert(0, ["2 RPR", m2.group(1)])
    if len(rows) < 2:
        return None
    deduped: list[list[str]] = []
    seen: set[tuple[str, str]] = set()
    for r in rows:
        key = (r[0], r[1])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    return PaperTable(
        caption="DockQ vs restraint count" if en else "DockQ 与约束数量",
        headers=["Restraint" if en else "约束", "Mean DockQ" if en else "DockQ"],
        rows=deduped[:6],
        table_num=0,
        source="restraint_heuristic",
    )


def _llm_extract_table(text: str, llm: LLMClient, language: str) -> PaperTable | None:
    system = (
        "Extract ONE key experimental results table from the paper for an academic poster. "
        "Prefer benchmark comparisons with numeric metrics (DockQ, success rate, RMSD, etc.). "
        "Output JSON: {caption, headers:[...], rows:[[...], ...]} with 3-6 rows max. "
        "Use ONLY numbers and labels from the paper. "
        + language_instruction(language)
    )
    preview = text[:14000]
    try:
        data = llm.chat_json(system, f"Paper excerpt:\n{preview}", temperature=0.2, max_tokens=2048)
        if not isinstance(data, dict):
            return None
        headers = data.get("headers") or []
        rows = data.get("rows") or []
        if len(headers) < 2 or len(rows) < 2:
            return None
        return PaperTable(
            caption=str(data.get("caption", "Key results"))[:120],
            headers=[str(h) for h in headers][:5],
            rows=[[str(c) for c in row][:5] for row in rows[:6]],
            table_num=0,
            source="llm",
        )
    except Exception:
        return None


def _row_has_number(cells: list[str]) -> bool:
    return any(re.search(r"\d+\.?\d*", c) for c in cells)


def _duplicate_table(existing: list[PaperTable], candidate: PaperTable) -> bool:
    sig = (candidate.caption[:40], len(candidate.rows))
    for t in existing:
        if (t.caption[:40], len(t.rows)) == sig:
            return True
    return False
