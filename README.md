# Model Cost/Quality Benchmark

**Do self-hosted open-weight models actually save money without giving up too
much quality? Measured, not assumed.**

## Why this exists

At Trinnix I had to compare API models against self-hosted ones to keep
inference costs under control without breaking SLAs on urgent tickets. This
repo is the general version of that decision so I can quickly reproduce the
same tradeoff analysis for any new support ticket classifier instead of
guessing.

## The question

For a fixed task — **classifying short support tickets into urgency tiers
(P1/P2/P3)** — how do these compare:

| | Rules baseline | Mock Claude Haiku | Mock Claude Sonnet | Mock Llama 3.1 8B | Mock Qwen 2.5 14B |
|---|---|---|---|---|---|
| Accuracy / eval score | 98.0% | 88.5% | 95.5% | 80.5% | 84.0% |
| Cost per 1,000 requests | $0.00 | $0.16 | $0.59 | $0.56 | $0.91 |
| p50 latency | 0 ms | 483 ms | 870 ms | 352 ms | 531 ms |
| p95 latency | 0 ms | 483 ms | 870 ms | 352 ms | 531 ms |
| Infra cost (if self-hosted) | — | — | — | $0.50/hr | $0.50/hr |

Full run output: [`docs/results.md`](docs/results.md). Enable real API/Ollama
providers in `configs/task.yaml` when you have keys or a local inference server.

## Method

- **Task**: Given a short support ticket (subject + 1–3 sentence description),
  predict an urgency tier (P1/P2/P3) that matches how a human triage process
  would route it.
- **Dataset**: 200 labelled synthetic-but-realistic tickets in
  [`evals/dataset.jsonl`](evals/dataset.jsonl), generated from common support
  patterns and manually checked for label consistency. Regenerate with
  `python evals/generate_dataset.py`.
- **Models tested**:
  - `rules-baseline` — deterministic keyword classifier (no API key).
  - `mock-*` profiles — reproducible harness validation with configured
    accuracy/latency targets; map to real providers via `configs/task.yaml`.
  - Real providers (Anthropic, OpenAI, Ollama) are wired but disabled by default.
- **Hosting**: API models via Anthropic/OpenAI SDKs; self-hosted via Ollama
  (`http://localhost:11434`). Mock self-hosted profiles use a reference GPU
  hourly rate for cost normalisation.
- **Cost methodology**: See [`docs/methodology.md`](docs/methodology.md). API
  costs use published per-token pricing in `configs/pricing.yaml`. Self-hosted
  costs estimate amortised cloud GPU time from measured tokens/sec.

## Results

On the 200-example synthetic eval set (run `20260708T111520Z`):

- **Rules baseline hit 98% accuracy** on this corpus because the synthetic
  tickets were built from the same urgency patterns the rules encode. That is
  useful as a harness sanity check, not as proof that keywords beat LLMs on
  messy production tickets.
- **Mock Sonnet profile (95.5%) vs mock Haiku (88.5%)** shows the quality gap
  you would expect when paying ~3.7x more per 1k requests ($0.59 vs $0.16).
- **Mock self-hosted models** land in a similar cost band to Haiku ($0.56–$0.91
  per 1k at the reference GPU rate) but with lower accuracy (80–84%). The
  tradeoff only wins if your real self-hosted throughput is materially higher
  than the conservative defaults, or if traffic volume amortises fixed GPU cost.

To run against real models: set `enabled: true` on `claude-haiku-3-5`,
`claude-sonnet-4`, or `llama-3.1-8b` in `configs/task.yaml` and export
`ANTHROPIC_API_KEY` / start Ollama.

## What I'd do differently at scale

At 10x traffic I would batch requests, cache repeated ticket patterns, and
re-run this benchmark monthly as pricing and quantised model releases change.
For production triage I would also track **P1 recall** separately from overall
accuracy, because missing a critical ticket is far worse than over-escalating.

## Reproduce it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python evals/generate_dataset.py
PYTHONPATH=src python -m benchmark.run --config configs/task.yaml
PYTHONPATH=src python -m benchmark.report results/ --output docs/results.md
pytest
```

## Repo structure

```
src/benchmark/   harness: model callers, cost calculators, runner, report
configs/         task + pricing configuration
evals/           labelled dataset + generator
docs/            methodology notes and benchmark results
tests/           unit tests for scoring and rules baseline
```
