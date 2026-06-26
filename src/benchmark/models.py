from __future__ import annotations

import hashlib
import os
import random
import re
import time
from dataclasses import dataclass

import httpx

from benchmark.config import ModelConfig


@dataclass
class ModelResponse:
    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    error: str | None = None


P1_KEYWORDS = (
    "outage",
    "down",
    "breach",
    "hacked",
    "data loss",
    "cannot login",
    "production",
    "all users",
    "security incident",
    "ransomware",
    "site is down",
    "complete failure",
)
P2_KEYWORDS = (
    "billing",
    "payment failed",
    "sla",
    "enterprise",
    "major",
    "broken",
    "degraded",
    "not working",
    "error 500",
    "timeout",
    "key customer",
    "revenue",
)


def _normalize_label(text: str) -> str:
    match = re.search(r"\bP[123]\b", text.upper())
    return match.group(0) if match else text.strip()


def rules_classify(subject: str, description: str) -> str:
    text = f"{subject} {description}".lower()
    if any(keyword in text for keyword in P1_KEYWORDS):
        return "P1"
    if any(keyword in text for keyword in P2_KEYWORDS):
        return "P2"
    return "P3"


def _mock_should_be_correct(example_id: str, model_id: str, target_accuracy: float) -> bool:
    digest = hashlib.sha256(f"{model_id}:{example_id}".encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < target_accuracy


def _mock_latency(model: ModelConfig) -> float:
    rng = random.Random(model.id)
    p50 = model.latency_ms_p50 or 400
    p95 = model.latency_ms_p95 or 800
    if rng.random() < 0.95:
        return rng.uniform(p50 * 0.85, p50 * 1.15)
    return rng.uniform(p50, p95)


class ModelRunner:
    def __init__(self, model: ModelConfig) -> None:
        self.model = model

    def run(self, prompt: str, *, example_id: str, expected_label: str) -> ModelResponse:
        provider = self.model.provider
        if provider == "rules":
            return self._run_rules(prompt)
        if provider == "mock":
            return self._run_mock(prompt, example_id=example_id, expected_label=expected_label)
        if provider == "anthropic":
            return self._run_anthropic(prompt)
        if provider == "openai":
            return self._run_openai(prompt)
        if provider == "ollama":
            return self._run_ollama(prompt)
        raise ValueError(f"Unsupported provider: {provider}")

    def _run_rules(self, prompt: str) -> ModelResponse:
        start = time.perf_counter()
        subject, description = _parse_prompt_fields(prompt)
        label = rules_classify(subject, description)
        latency_ms = (time.perf_counter() - start) * 1000
        return ModelResponse(text=label, latency_ms=latency_ms, input_tokens=0, output_tokens=0)

    def _run_mock(self, prompt: str, *, example_id: str, expected_label: str) -> ModelResponse:
        time.sleep((_mock_latency(self.model) / 1000) * 0.01)
        subject, description = _parse_prompt_fields(prompt)
        baseline = rules_classify(subject, description)
        target = self.model.target_accuracy or 0.85
        if _mock_should_be_correct(example_id, self.model.id, target):
            label = expected_label
        else:
            label = _wrong_label(expected_label, baseline)
        return ModelResponse(
            text=label,
            latency_ms=_mock_latency(self.model),
            input_tokens=self.model.input_tokens or 180,
            output_tokens=self.model.output_tokens or 3,
        )

    def _run_anthropic(self, prompt: str) -> ModelResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return ModelResponse("", 0, 0, 0, error="ANTHROPIC_API_KEY not set")
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=api_key)
            start = time.perf_counter()
            response = client.messages.create(
                model=self.model.model or "claude-3-5-haiku-latest",
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.content[0].text if response.content else ""
            usage = response.usage
            return ModelResponse(
                text=_normalize_label(text),
                latency_ms=latency_ms,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResponse("", 0, 0, 0, error=str(exc))

    def _run_openai(self, prompt: str) -> ModelResponse:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return ModelResponse("", 0, 0, 0, error="OPENAI_API_KEY not set")
        try:
            from openai import OpenAI

            client = OpenAI(api_key=api_key)
            start = time.perf_counter()
            response = client.chat.completions.create(
                model=self.model.model or "gpt-4o-mini",
                max_tokens=16,
                messages=[{"role": "user", "content": prompt}],
            )
            latency_ms = (time.perf_counter() - start) * 1000
            text = response.choices[0].message.content or ""
            usage = response.usage
            return ModelResponse(
                text=_normalize_label(text),
                latency_ms=latency_ms,
                input_tokens=usage.prompt_tokens if usage else 0,
                output_tokens=usage.completion_tokens if usage else 0,
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResponse("", 0, 0, 0, error=str(exc))

    def _run_ollama(self, prompt: str) -> ModelResponse:
        base_url = self.model.base_url or "http://localhost:11434"
        try:
            start = time.perf_counter()
            with httpx.Client(timeout=120.0) as client:
                response = client.post(
                    f"{base_url}/api/generate",
                    json={"model": self.model.model, "prompt": prompt, "stream": False},
                )
                response.raise_for_status()
                payload = response.json()
            latency_ms = (time.perf_counter() - start) * 1000
            text = payload.get("response", "")
            eval_count = payload.get("eval_count", 3)
            prompt_eval_count = payload.get("prompt_eval_count", 180)
            return ModelResponse(
                text=_normalize_label(text),
                latency_ms=latency_ms,
                input_tokens=prompt_eval_count,
                output_tokens=eval_count,
            )
        except Exception as exc:  # noqa: BLE001
            return ModelResponse("", 0, 0, 0, error=str(exc))


def _parse_prompt_fields(prompt: str) -> tuple[str, str]:
    subject_match = re.search(r"Subject:\s*(.+)", prompt)
    description_match = re.search(r"Description:\s*(.+)", prompt, re.DOTALL)
    subject = subject_match.group(1).strip() if subject_match else ""
    description = description_match.group(1).strip() if description_match else prompt
    return subject, description


def _wrong_label(expected: str, baseline: str) -> str:
    options = ["P1", "P2", "P3"]
    for candidate in (baseline, expected):
        remaining = [label for label in options if label != candidate]
        if remaining:
            return remaining[0]
    return "P3"
