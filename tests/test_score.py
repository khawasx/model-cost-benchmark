from benchmark.score import percentile, score_predictions


def test_score_predictions_perfect():
    rows = [
        {"expected_label": "P1", "predicted_label": "P1"},
        {"expected_label": "P2", "predicted_label": "P2"},
        {"expected_label": "P3", "predicted_label": "P3"},
    ]
    metrics = score_predictions(rows)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_percentile():
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert percentile(values, 50) == 30.0
