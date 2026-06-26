# Benchmark results

Run directory: `results/20260708T111520Z`

| Model              | Provider   | Accuracy   | Macro F1   | Cost / 1k req (USD)   |   p50 latency (ms) |   p95 latency (ms) | Infra $/hr   |
|--------------------|------------|------------|------------|-----------------------|--------------------|--------------------|--------------|
| rules-baseline     | rules      | 98.0%      | 98.0%      | $0.0000               |                  0 |                  0 | —            |
| mock-claude-haiku  | mock       | 88.5%      | 88.6%      | $0.1560               |                483 |                483 | —            |
| mock-claude-sonnet | mock       | 95.5%      | 95.5%      | $0.5850               |                870 |                870 | —            |
| mock-llama-3.1-8b  | mock       | 80.5%      | 81.1%      | $0.5648               |                352 |                352 | $0.50        |
| mock-qwen-2.5-14b  | mock       | 84.0%      | 84.3%      | $0.9077               |                531 |                531 | $0.50        |
