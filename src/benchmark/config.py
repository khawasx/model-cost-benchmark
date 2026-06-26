from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    name: str
    description: str
    labels: list[str]
    prompt_template: str


class ModelConfig(BaseModel):
    id: str
    provider: str
    enabled: bool = True
    model: str | None = None
    pricing_key: str | None = None
    base_url: str | None = None
    self_hosted: bool = False
    tokens_per_second: float | None = None
    target_accuracy: float | None = None
    latency_ms_p50: float | None = None
    latency_ms_p95: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None


class BenchmarkConfig(BaseModel):
    task: TaskConfig
    models: list[ModelConfig]


class PricingConfig(BaseModel):
    api_models: dict[str, dict[str, float]] = Field(default_factory=dict)
    self_hosted: dict[str, float] = Field(default_factory=dict)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return BenchmarkConfig.model_validate(data)


def load_pricing_config(path: Path) -> PricingConfig:
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return PricingConfig.model_validate(data)


def load_dataset(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
