"""DeepSeek OpenAI 兼容 API 客户端."""

from __future__ import annotations

import json
import re
from typing import Any

from openai import OpenAI

from poster_agent.config import LLMConfig


class LLMClient:
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
        )
        return response.choices[0].message.content or ""

    def chat_json(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        retries: int = 2,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            prompt = user
            if attempt > 0:
                prompt += "\n\n上次输出不是合法 JSON，请重新输出完整且可解析的 JSON。"
            text = self.chat(
                system + "\n请只输出合法 JSON，不要包含 markdown 代码块。",
                prompt,
                temperature=temperature,
            )
            try:
                return _parse_json(text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise last_error  # type: ignore[misc]


def _parse_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
