from __future__ import annotations

from typing import Any

from benchmark.config import ModelConfig, PricingConfig


def cost_per_thousand_requests(
    *,
    model: ModelConfig,
    pricing: PricingConfig,
    avg_input_tokens: float,
    avg_output_tokens: float,
    avg_latency_ms: float,
) -> tuple[float, float | None]:
    """Return (cost_per_1000_requests_usd, infra_hourly_usd_if_self_hosted)."""
    if model.provider == "rules":
        return 0.0, None

    if model.provider == "mock" and not model.self_hosted:
        pass
    elif model.self_hosted:
        tokens_per_second = model.tokens_per_second or max(
            (avg_input_tokens + avg_output_tokens) / max(avg_latency_ms / 1000, 0.001),
            1.0,
        )
        gpu_hourly = pricing.self_hosted.get("gpu_hourly_usd", 0.50)
        seconds_per_request = (avg_input_tokens + avg_output_tokens) / tokens_per_second
        cost_per_request = (gpu_hourly / 3600) * seconds_per_request
        return cost_per_request * 1000, gpu_hourly

    pricing_key = model.pricing_key or model.id
    if model.model and model.model in pricing.api_models:
        pricing_key = model.model
    rates = pricing.api_models.get(pricing_key) or pricing.api_models.get(model.id)
    if not rates:
        return 0.0, None

    input_cost = (avg_input_tokens / 1_000_000) * rates["input_per_million"]
    output_cost = (avg_output_tokens / 1_000_000) * rates["output_per_million"]
    return (input_cost + output_cost) * 1000, None


def aggregate_usage(rows: list[dict[str, Any]]) -> dict[str, float]:
    valid = [row for row in rows if not row.get("error")]
    if not valid:
        return {"avg_input_tokens": 0.0, "avg_output_tokens": 0.0, "avg_latency_ms": 0.0}
    return {
        "avg_input_tokens": sum(row["input_tokens"] for row in valid) / len(valid),
        "avg_output_tokens": sum(row["output_tokens"] for row in valid) / len(valid),
        "avg_latency_ms": sum(row["latency_ms"] for row in valid) / len(valid),
    }
