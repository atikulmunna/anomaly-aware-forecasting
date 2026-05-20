from aaf.experiments.compare import flatten_mapping


def test_flatten_mapping_uses_dotted_metric_keys() -> None:
    flattened = flatten_mapping(
        {
            "forecast": {"nll": 1.5},
            "anomaly": {"threshold_free": {"vus_pr": 0.8}},
        }
    )

    assert flattened == {
        "forecast.nll": 1.5,
        "anomaly.threshold_free.vus_pr": 0.8,
    }
