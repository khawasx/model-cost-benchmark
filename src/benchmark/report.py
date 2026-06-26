from __future__ import annotations

import json
from pathlib import Path

import typer
from tabulate import tabulate

app = typer.Typer(add_completion=False)


def _latest_run_dir(results_dir: Path) -> Path:
    runs = sorted([path for path in results_dir.iterdir() if path.is_dir()])
    if not runs:
        raise FileNotFoundError(f"No benchmark runs found in {results_dir}")
    return runs[-1]


def build_markdown_table(summary: dict[str, dict]) -> str:
    headers = [
        "Model",
        "Provider",
        "Accuracy",
        "Macro F1",
        "Cost / 1k req (USD)",
        "p50 latency (ms)",
        "p95 latency (ms)",
        "Infra $/hr",
    ]
    rows = []
    for model_id, stats in summary.items():
        rows.append(
            [
                model_id,
                stats["provider"],
                f"{stats['accuracy'] * 100:.1f}%",
                f"{stats['macro_f1'] * 100:.1f}%",
                f"${stats['cost_per_1000_usd']:.4f}",
                f"{stats['p50_latency_ms']:.0f}",
                f"{stats['p95_latency_ms']:.0f}",
                f"${stats['infra_hourly_usd']:.2f}" if stats.get("infra_hourly_usd") else "—",
            ]
        )
    return tabulate(rows, headers=headers, tablefmt="github")


@app.command()
def main(
    results: Path = typer.Argument(..., help="Results directory or specific run directory"),
    output: Path = typer.Option(
        Path("docs/results.md"),
        "--output",
        help="Markdown report output path",
    ),
) -> None:
    """Generate a markdown report from a benchmark run summary."""
    run_dir = results if (results / "summary.json").exists() else _latest_run_dir(results)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    table = build_markdown_table(summary)

    output.parent.mkdir(parents=True, exist_ok=True)
    content = f"# Benchmark results\n\nRun directory: `{run_dir}`\n\n{table}\n"
    output.write_text(content, encoding="utf-8")
    print(content)


if __name__ == "__main__":
    app()
