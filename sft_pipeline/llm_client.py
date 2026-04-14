from __future__ import annotations

import json
import time
from typing import Any, Protocol

from .config import RunConfig


class JSONClient(Protocol):
    def generate_json_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ...

    def close(self) -> None:
        ...


class LocalChatClient:
    def __init__(self, config: RunConfig) -> None:
        from openai import OpenAI

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
                parsed = _parse_model_json(content)
                return parsed, usage_totals
            except Exception as exc:
                last_error = exc
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not contain a valid JSON object. "
                            "Return the required JSON object only, with no reasoning and no code fences. "
                            "Escape every backslash correctly, for example write \\\\ instead of \\."
                        ),
                    }
                ]
        raise last_error or ValueError("Failed to parse JSON response")

    def generate_json_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        results: list[dict[str, Any]] = []
        usages: list[dict[str, Any]] = []
        for messages in messages_batch:
            result, usage = self.generate_json(messages, max_tokens=max_tokens)
            results.append(result)
            usages.append(usage)
        return results, usages

    def close(self) -> None:
        return None


class InProcessVLLMClient:
    def __init__(self, config: RunConfig) -> None:
        from transformers import AutoTokenizer
        from vllm import LLM

        self.config = config
        # Gemma 4 ships a tokenizer_config whose extra_special_tokens field is a list.
        # Current transformers loading is stricter here, so we override it to a mapping.
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model,
            trust_remote_code=config.trust_remote_code,
            extra_special_tokens={},
        )
        self.llm = LLM(
            model=config.model,
            tokenizer=config.model,
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            max_num_seqs=config.max_num_seqs,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enforce_eager=config.enforce_eager,
            trust_remote_code=config.trust_remote_code,
            dtype=config.dtype,
        )

    def generate_json(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        results, usages = self.generate_json_batch([messages], max_tokens=max_tokens)
        return results[0], usages[0]

    def generate_json_batch(
        self,
        messages_batch: list[list[dict[str, str]]],
        max_tokens: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        from vllm import SamplingParams

        results: list[dict[str, Any] | None] = [None] * len(messages_batch)
        usages: list[dict[str, Any]] = [
            {"latency_sec": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            for _ in messages_batch
        ]
        pending: dict[int, list[dict[str, str]]] = {idx: messages for idx, messages in enumerate(messages_batch)}
        last_error: Exception | None = None

        for _ in range(3):
            if not pending:
                break
            prompt_indices = list(pending.keys())
            prompts = [self._render_messages(pending[idx]) for idx in prompt_indices]
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                stop=list(self.config.stop_strings),
            )
            started = time.perf_counter()
            outputs = self.llm.generate(prompts, sampling_params=sampling_params)
            elapsed = time.perf_counter() - started
            per_request_latency = elapsed / max(len(outputs), 1)
            retry_messages: dict[int, list[dict[str, str]]] = {}

            for idx, output in zip(prompt_indices, outputs):
                primary = output.outputs[0] if output.outputs else None
                text = primary.text if primary else ""
                prompt_tokens = len(output.prompt_token_ids or [])
                completion_tokens = len(primary.token_ids or []) if primary else 0
                usage = usages[idx]
                usage["latency_sec"] += per_request_latency
                usage["prompt_tokens"] += prompt_tokens
                usage["completion_tokens"] += completion_tokens
                usage["total_tokens"] += prompt_tokens + completion_tokens
                try:
                    results[idx] = _parse_model_json(text)
                except Exception as exc:
                    last_error = exc
                    retry_messages[idx] = messages_batch[idx] + [
                        {
                            "role": "user",
                            "content": (
                                "Your previous answer did not contain a valid JSON object. "
                                "Return the required JSON object only, with no reasoning and no code fences. "
                                "Escape every backslash correctly, for example write \\\\ instead of \\."
                            ),
                        }
                    ]
            pending = retry_messages

        unresolved = [idx for idx, result in enumerate(results) if result is None]
        for idx in unresolved:
            try:
                result, usage = self._generate_json_single_with_retries(messages_batch[idx], max_tokens=max_tokens)
                results[idx] = result
                merged_usage = usages[idx]
                for key, value in usage.items():
                    merged_usage[key] = merged_usage.get(key, 0) + value
            except Exception as exc:
                last_error = exc

        unresolved = [idx for idx, result in enumerate(results) if result is None]
        if unresolved:
            raise last_error or ValueError(f"Failed to parse JSON for items: {unresolved}")
        return [result for result in results if result is not None], usages

    def close(self) -> None:
        return None

    def _render_messages(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def _generate_json_single_with_retries(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from vllm import SamplingParams

        attempt_messages = list(messages)
        usage = {"latency_sec": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        last_error: Exception | None = None

        for _ in range(4):
            prompt = self._render_messages(attempt_messages)
            sampling_params = SamplingParams(
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                stop=list(self.config.stop_strings),
            )
            started = time.perf_counter()
            outputs = self.llm.generate([prompt], sampling_params=sampling_params)
            elapsed = time.perf_counter() - started
            output = outputs[0]
            primary = output.outputs[0] if output.outputs else None
            text = primary.text if primary else ""
            prompt_tokens = len(output.prompt_token_ids or [])
            completion_tokens = len(primary.token_ids or []) if primary else 0
            usage["latency_sec"] += elapsed
            usage["prompt_tokens"] += prompt_tokens
            usage["completion_tokens"] += completion_tokens
            usage["total_tokens"] += prompt_tokens + completion_tokens
            try:
                return _parse_model_json(text), usage
            except Exception as exc:
                last_error = exc
                attempt_messages = messages + [
                    {
                        "role": "user",
                        "content": (
                            "Your previous answer did not contain a valid JSON object. "
                            "Return valid JSON only. Do not use markdown code fences. "
                            "Escape every backslash correctly, for example write \\\\ instead of \\."
                        ),
                    }
                ]
        raise last_error or ValueError("Failed to parse JSON response")


def build_json_client(config: RunConfig) -> JSONClient:
    if config.backend == "openai_api":
        return LocalChatClient(config)
    return InProcessVLLMClient(config)


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


def _parse_model_json(text: str) -> dict[str, Any]:
    candidate = _extract_json_object(text)
    parse_attempts = [
        candidate,
        _escape_invalid_backslashes(candidate),
        _remove_trailing_commas(_escape_invalid_backslashes(candidate)),
    ]
    last_error: Exception | None = None
    for attempt in parse_attempts:
        try:
            parsed = json.loads(attempt)
            if not isinstance(parsed, dict):
                raise ValueError("Model output JSON is not an object")
            return parsed
        except Exception as exc:
            last_error = exc
    raise last_error or ValueError("Failed to parse JSON response")


def _escape_invalid_backslashes(text: str) -> str:
    escaped: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
            escaped.append(ch)
            i += 1
            continue
        if ch == "\\" and in_string:
            next_ch = text[i + 1] if i + 1 < len(text) else ""
            if next_ch not in {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}:
                escaped.append("\\\\")
                i += 1
                continue
        escaped.append(ch)
        i += 1
    return "".join(escaped)


def _remove_trailing_commas(text: str) -> str:
    cleaned: list[str] = []
    in_string = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            backslash_count = 0
            j = i - 1
            while j >= 0 and text[j] == "\\":
                backslash_count += 1
                j -= 1
            if backslash_count % 2 == 0:
                in_string = not in_string
        if not in_string and ch == ",":
            j = i + 1
            while j < len(text) and text[j].isspace():
                j += 1
            if j < len(text) and text[j] in "}]":
                i += 1
                continue
        cleaned.append(ch)
        i += 1
    return "".join(cleaned)
