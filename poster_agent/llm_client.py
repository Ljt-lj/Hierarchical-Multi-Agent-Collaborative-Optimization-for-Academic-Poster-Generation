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
        max_tokens: int | None = None,
        retries: int = 3,
    ) -> Any:
        last_error: Exception | None = None
        for attempt in range(retries + 1):
            prompt = user
            if attempt > 0:
                prompt += "\n\nYour previous output was empty or invalid JSON. Reply with ONLY valid JSON."
            text = self.chat(
                system + "\nOutput ONLY valid JSON. No markdown fences, no commentary.",
                prompt,
                temperature=temperature if temperature is not None else max(0.1, (self.config.temperature or 0.3) - 0.1 * attempt),
                max_tokens=max_tokens,
            )
            if not text.strip():
                last_error = ValueError("LLM returned empty response")
                continue
            try:
                return _parse_json(text)
            except (json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"Failed to parse LLM JSON after {retries + 1} attempts: {last_error}") from last_error


def _parse_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("empty JSON text")
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
