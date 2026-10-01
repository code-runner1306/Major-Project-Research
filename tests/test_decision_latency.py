from decision_evaluation import summarize_latency


def test_latency_summary_groups_primitive_length_and_option_count():
    rows = [
        {"primitive": "choice", "serialized_length": 80, "option_count": 2, "latency_seconds": 0.01},
        {"primitive": "choice", "serialized_length": 300, "option_count": 4, "latency_seconds": 0.03},
        {"primitive": "score", "serialized_length": 450, "option_count": 5, "latency_seconds": 0.05},
    ]

    summary = summarize_latency(rows)

    assert summary["overall"]["decisions"] == 3
    assert summary["primitive"]["choice"]["decisions"] == 2
    assert summary["serialized_length_bucket"]["001-128"]["decisions"] == 1
    assert summary["serialized_length_bucket"]["257-512"]["decisions"] == 2
    assert summary["option_count"]["5"]["decisions"] == 1
    assert summary["primitive_x_length_bucket"]["score/257-512"]["p95_seconds"] is not None


def test_latency_summary_explicitly_counts_unmeasured_rows():
    summary = summarize_latency([
        {"primitive": "noul", "serialized_length": 20, "option_count": 2},
    ])

    assert summary["overall"]["decisions"] == 0
    assert summary["missing_latency_records"] == 1
