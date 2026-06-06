"""Refiner JSON 解析与 P2P 格式 → ContentNode 转换."""

from __future__ import annotations

import json
import re
from typing import Any

from poster_agent.llm_client import _parse_json
from poster_agent.models.trees import ContentNode

CONTENT_KEYS = {"title", "summary", "bullets", "weight", "logic_links", "children"}


def strip_markdown_fences(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        return fence.group(1).strip()
    return text


def balanced_json_slice(text: str) -> str | None:
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_bullets(text: str, limit: int = 4) -> list[str]:
    bullets: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(("-", "•", "*")):
            bullets.append(stripped.lstrip("-•* ").strip())
        elif stripped and len(stripped) < 200 and stripped[0].isdigit() and "." in stripped[:4]:
            bullets.append(stripped)
    if bullets:
        return bullets[:limit]
    parts = [s.strip() for s in text.replace("\n", " ").split(". ") if len(s.strip()) > 10]
    return parts[:limit] if parts else ([text[:120]] if text.strip() else [""])


def markdown_sections_to_tree(text: str) -> dict[str, Any] | None:
    sections = re.split(r"\n(?=##\s+)", text.strip())
    if len(sections) <= 1 and not text.strip().startswith("##"):
        return None

    children: list[dict[str, Any]] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        match = re.match(r"##\s+(.+?)(?:\n|$)", sec)
        if match:
            title = match.group(1).strip()
            body = sec[match.end() :].strip()
        else:
            title = "Section"
            body = sec
        bullets = extract_bullets(body)
        children.append(
            {
                "title": title,
                "summary": body[:120],
                "bullets": bullets[:4],
                "weight": 0.5,
                "logic_links": [],
                "children": [],
            }
        )
    if not children:
        return None
    return {
        "title": children[0]["title"],
        "summary": children[0]["summary"],
        "bullets": children[0]["bullets"][:3],
        "weight": 1.0,
        "logic_links": [],
        "children": children[1:] if len(children) > 1 else [],
    }


def _safe_weight(raw: Any, default: float = 0.5) -> float:
    try:
        return float(raw) if raw is not None else default
    except (TypeError, ValueError):
        return default


def p2p_dict_to_content_tree(obj: dict[str, Any]) -> dict[str, Any]:
    if CONTENT_KEYS.intersection(obj.keys()):
        return {
            "title": str(obj.get("title") or "Poster"),
            "summary": str(obj.get("summary") or "")[:120],
            "bullets": [str(b) for b in (obj.get("bullets") or [])[:4]],
            "weight": _safe_weight(obj.get("weight"), 1.0),
            "logic_links": [str(x) for x in (obj.get("logic_links") or [])],
            "children": [
                p2p_dict_to_content_tree(c) for c in (obj.get("children") or []) if isinstance(c, dict)
            ],
        }

    children: list[dict[str, Any]] = []
    for key, val in obj.items():
        if key.lower() in {"title", "summary", "metadata"}:
            continue
        if isinstance(val, dict):
            content = val.get("content") or val.get("text") or val.get("summary") or json.dumps(val, ensure_ascii=False)
        else:
            content = str(val)
        content = str(content).strip()
        children.append(
            {
                "title": str(key),
                "summary": content[:120],
                "bullets": extract_bullets(content)[:4],
                "weight": 0.5,
                "logic_links": [],
                "children": [],
            }
        )
    if not children:
        return {
            "title": "Poster",
            "summary": "",
            "bullets": [],
            "weight": 1.0,
            "logic_links": [],
            "children": [],
        }
    return {
        "title": "Poster",
        "summary": children[0]["summary"],
        "bullets": children[0]["bullets"][:3],
        "weight": 1.0,
        "logic_links": [],
        "children": children,
    }


def try_parse_json_obj(text: str) -> Any | None:
    text = strip_markdown_fences(text)
    for candidate in (text, balanced_json_slice(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def to_content_tree_dict(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    obj = try_parse_json_obj(text)
    if isinstance(obj, dict):
        return p2p_dict_to_content_tree(obj)
    if "##" in text or re.search(r"^#+\s", text, re.MULTILINE):
        return markdown_sections_to_tree(text)
    if len(text) >= 200:
        return {
            "title": "Poster",
            "summary": text[:120],
            "bullets": extract_bullets(text)[:4],
            "weight": 1.0,
            "logic_links": [],
            "children": [],
        }
    return None


def normalize_refiner_json(text: str, user: str = "") -> str | None:
    tree = to_content_tree_dict(text)
    if tree is None:
        return None
    try:
        ContentNode.from_dict(tree)
    except Exception:
        return None
    return json.dumps(tree, ensure_ascii=False)


def load_content_node_from_text(text: str, *, allow_repair: bool = True) -> ContentNode:
    obj = try_parse_json_obj(text)
    if obj is None:
        try:
            obj = _parse_json(text)
        except (json.JSONDecodeError, ValueError):
            obj = None
    if isinstance(obj, dict):
        return ContentNode.from_dict(p2p_dict_to_content_tree(obj))
    if allow_repair:
        repaired = normalize_refiner_json(text)
        if repaired:
            obj = json.loads(repaired)
            return ContentNode.from_dict(p2p_dict_to_content_tree(obj))
    raise ValueError("refiner output is not valid JSON / ContentNode")


def parse_error(text: str) -> str | None:
    try:
        load_content_node_from_text(text, allow_repair=False)
        return None
    except Exception as exc:
        repaired = normalize_refiner_json(text)
        if repaired:
            try:
                load_content_node_from_text(repaired, allow_repair=False)
                return f"repairable: {exc}"
            except Exception as exc2:
                return f"{exc}; after repair: {exc2}"
        return str(exc)


def load_content_node_from_dict(data: dict[str, Any]) -> ContentNode:
    return ContentNode.from_dict(p2p_dict_to_content_tree(data))


def is_parseable_content_json(text: str) -> bool:
    try:
        load_content_node_from_text(text)
        return True
    except Exception:
        return False
