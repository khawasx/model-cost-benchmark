# Cost methodology

## API models

Cost per request is computed from measured input/output token counts and the
published per-million token rates in `configs/pricing.yaml`.

```
cost_per_request = (input_tokens / 1e6) * input_rate + (output_tokens / 1e6) * output_rate
cost_per_1000 = cost_per_request * 1000
```

## Self-hosted models

Self-hosted cost is estimated from throughput and a configurable GPU hourly
rate (default: $0.50/hr, a typical cloud L4/A10 reference price):

```
seconds_per_request = (input_tokens + output_tokens) / tokens_per_second
cost_per_request = (gpu_hourly_usd / 3600) * seconds_per_request
```

This intentionally ignores local Mac amortisation so API vs cloud-GPU
self-hosting comparisons stay consistent. Document your own hardware economics
separately if running on owned hardware.

## Rules baseline

The `rules-baseline` provider has zero token cost and near-zero latency. It
exists as a reproducible floor and sanity check for the harness.
