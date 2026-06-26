from benchmark.models import rules_classify


def test_rules_classify_p1():
    assert rules_classify("Production outage", "Site is down for all users") == "P1"


def test_rules_classify_p2():
    assert rules_classify("Billing issue", "Enterprise customer reports duplicate charges") == "P2"


def test_rules_classify_p3():
    assert rules_classify("Feature request", "Please add CSV export") == "P3"
