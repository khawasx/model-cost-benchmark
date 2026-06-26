"""Generate synthetic support ticket dataset for urgency classification."""

from __future__ import annotations

import json
import random
from pathlib import Path

TEMPLATES = {
    "P1": [
        ("Production outage", "The entire production site is down and all users cannot login."),
        ("Security breach", "We detected a security breach and suspect customer data exposure."),
        ("Data loss", "Database restore failed and we may have permanent data loss on orders."),
        ("Complete failure", "Payment processing is completely down for all merchants."),
        ("Ransomware alert", "Ransomware detected on production servers; systems isolated."),
        ("Global outage", "API returns 503 for all regions; status page confirms major outage."),
    ],
    "P2": [
        ("Billing error", "Enterprise customer reports duplicate charges on last invoice."),
        ("Major feature broken", "Checkout is broken for premium users; revenue impact expected."),
        ("SLA risk", "Key customer SLA breach risk due to repeated timeout errors."),
        ("Payment failed", "Subscription renewals failing for many accounts since this morning."),
        ("Degraded performance", "Dashboard loading is severely degraded for enterprise tenants."),
        ("Error 500 spike", "Error 500 rate increased 8x on order service for paid plans."),
    ],
    "P3": [
        ("Feature request", "Can you add CSV export for monthly reports?"),
        ("Minor UI bug", "Button alignment is off on settings page in Safari."),
        ("How-to question", "How do I invite teammates to my workspace?"),
        ("Cosmetic issue", "Logo looks blurry on retina displays but functionality works."),
        ("Docs clarification", "Documentation unclear on webhook retry policy."),
        ("Low impact bug", "Tooltip text has a typo on the profile page."),
    ],
}

VARIATIONS = [
    "Customer says: {text}",
    "Reported by support queue: {text}",
    "Ticket notes: {text}",
    "User message: {text}",
    "Escalation summary: {text}",
]


def generate_dataset(count: int = 200, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    per_label = count // 3
    remainder = count % 3
    label_counts = {"P1": per_label, "P2": per_label, "P3": per_label}
    for index, label in enumerate(["P1", "P2", "P3"]):
        if index < remainder:
            label_counts[label] += 1

    rows: list[dict] = []
    ticket_id = 1
    for label, target_count in label_counts.items():
        templates = TEMPLATES[label]
        for _ in range(target_count):
            subject, description = rng.choice(templates)
            variation = rng.choice(VARIATIONS).format(text=description)
            rows.append(
                {
                    "id": f"ticket-{ticket_id:04d}",
                    "subject": subject,
                    "description": variation,
                    "label": label,
                }
            )
            ticket_id += 1

    rng.shuffle(rows)
    return rows


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "evals" / "dataset.jsonl"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_dataset()
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} examples to {output}")


if __name__ == "__main__":
    main()
