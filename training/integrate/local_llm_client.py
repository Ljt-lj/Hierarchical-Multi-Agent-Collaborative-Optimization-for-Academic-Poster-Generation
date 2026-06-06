"""本地微调模型客户端：与 poster_agent.llm_client.LLMClient 接口兼容."""

from __future__ import annotations

import json
import re
from typing import Any

from poster_agent.llm_client import _parse_json


class LocalLoRAClient:
    """加载 Qwen + LoRA adapter，供 Refiner/Visual Agent 调用."""

    def __init__(
        self,
        base_model: str,
        adapter_path: str,
        *,
        max_new_tokens: int = 2048,
        temperature: float = 0.3,
    ):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        temp = temperature if temperature is not None else self.temperature
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.max_new_tokens,
                do_sample=temp > 0,
                temperature=max(temp, 0.01),
            )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

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
