from __future__ import annotations

import json
import time
from typing import Any

from openai import OpenAI

from .config import RunConfig


class LocalChatClient:
    def __init__(self, config: RunConfig) -> None:
        self.config = config
        self.client = OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            timeout=config.timeout,
            default_headers=config.extra_headers or None,
        )

    def generate_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        attempt_messages = list(messages)
        usage_totals = {"latency_sec": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error: Exception | None = None
        for attempt in range(3):
            started = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=attempt_messages,
                temperature=self.config.temperature,
                max_tokens=max_tokens,
            )
            elapsed = time.perf_counter() - started
            usage_totals["latency_sec"] += elapsed
            usage_totals["prompt_tokens"] += int(getattr(response.usage, "prompt_tokens", 0) or 0)
            usage_totals["completion_tokens"] += int(getattr(response.usage, "completion_tokens", 0) or 0)
            usage_totals["total_tokens"] += int(getattr(response.usage, "total_tokens", 0) or 0)
            content = response.choices[0].message.content or ""
            try:
                parsed = json.loads(_extract_json_object(content))
                return parsed, usage_totals
            except Exception as exc:
                last_error = exc
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not contain a valid JSON object. "
                            "Return the required JSON object only, with no reasoning and no code fences."
                        ),
                    }
                ]
        raise last_error or ValueError("Failed to parse JSON response")


def _extract_json_object(text: str) -> str:
    stripped = text.strip()
    if "<think>" in stripped and "</think>" in stripped:
        stripped = stripped.split("</think>", 1)[1].strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        stripped = "\n".join(lines[1:-1]).strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model output does not contain JSON object: {text[:500]}")
    return stripped[start : end + 1]
