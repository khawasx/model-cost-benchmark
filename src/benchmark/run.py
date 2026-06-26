from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from benchmark.config import BenchmarkConfig, load_benchmark_config, load_dataset, load_pricing_config
from benchmark.cost import aggregate_usage, cost_per_thousand_requests
from benchmark.models import ModelRunner
from benchmark.score import parse_label, percentile, score_predictions

app = typer.Typer(add_completion=False)
console = Console()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _build_prompt(template: str, row: dict) -> str:
    return template.format(
        subject=row["subject"],
        description=row["description"],
    )


def run_benchmark(
    config: BenchmarkConfig,
    dataset: list[dict],
    pricing_path: Path,
    output_dir: Path,
) -> Path:
    pricing = load_pricing_config(pricing_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict] = {}
    enabled_models = [model for model in config.models if model.enabled]

    for model in enabled_models:
        console.print(f"[bold]Running[/bold] {model.id} ({model.provider})")
        runner = ModelRunner(model)
        predictions: list[dict] = []

        for row in dataset:
            prompt = _build_prompt(config.task.prompt_template, row)
            response = runner.run(
                prompt,
                example_id=row["id"],
                expected_label=row["label"],
            )
            predicted = parse_label(response.text) if not response.error else ""
            predictions.append(
                {
                    "id": row["id"],
                    "expected_label": row["label"],
                    "predicted_label": predicted,
                    "raw_output": response.text,
                    "latency_ms": response.latency_ms,
                    "input_tokens": response.input_tokens,
                    "output_tokens": response.output_tokens,
                    "error": response.error,
                }
            )

        model_path = run_dir / f"{model.id}.jsonl"
        with model_path.open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item) + "\n")

        scored_rows = [row for row in predictions if not row.get("error")]
        metrics = score_predictions(scored_rows) if scored_rows else {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "total": 0.0,
            "correct": 0.0,
        }
        usage = aggregate_usage(predictions)
        latencies = [row["latency_ms"] for row in scored_rows]
        cost_1k, infra_hourly = cost_per_thousand_requests(
            model=model,
            pricing=pricing,
            avg_input_tokens=usage["avg_input_tokens"],
            avg_output_tokens=usage["avg_output_tokens"],
            avg_latency_ms=usage["avg_latency_ms"],
        )

        summary[model.id] = {
            "provider": model.provider,
            "self_hosted": model.self_hosted,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "cost_per_1000_usd": cost_1k,
            "infra_hourly_usd": infra_hourly,
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "errors": sum(1 for row in predictions if row.get("error")),
            "avg_input_tokens": usage["avg_input_tokens"],
            "avg_output_tokens": usage["avg_output_tokens"],
        }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    console.print(f"[green]Wrote results to[/green] {run_dir}")
    return run_dir


@app.command()
def main(
    config: Path = typer.Option(
        _project_root() / "configs" / "task.yaml",
        "--config",
        help="Path to task config YAML",
    ),
    dataset: Path = typer.Option(
        _project_root() / "evals" / "dataset.jsonl",
        "--dataset",
        help="Path to labelled dataset JSONL",
    ),
    pricing: Path = typer.Option(
        _project_root() / "configs" / "pricing.yaml",
        "--pricing",
        help="Path to pricing config YAML",
    ),
    output: Path = typer.Option(
        _project_root() / "results",
        "--output",
        help="Directory for run outputs",
    ),
) -> None:
    """Run all enabled models against the labelled dataset."""
    benchmark_config = load_benchmark_config(config)
    rows = load_dataset(dataset)
    run_benchmark(benchmark_config, rows, pricing, output)


if __name__ == "__main__":
    app()
