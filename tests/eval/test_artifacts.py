import json

import numpy as np

from aaf.eval.artifacts import write_mixture_diagnostics_json, write_regime_diagnostics_json
from aaf.eval.forecasting import MixtureForecast


def test_write_mixture_diagnostics_json_writes_requested_splits(tmp_path) -> None:
    forecast = MixtureForecast.from_arrays(
        weights=np.ones((2, 1)),
        means=np.zeros((2, 1, 1)),
        stds=np.ones((2, 1, 1)),
    )

    write_mixture_diagnostics_json(tmp_path / "diagnostics.json", test=forecast)
    payload = json.loads((tmp_path / "diagnostics.json").read_text(encoding="utf-8"))

    assert set(payload) == {"test"}
    assert payload["test"]["active_components_1pct"] == 1


def test_write_regime_diagnostics_json_writes_summary(tmp_path) -> None:
    write_regime_diagnostics_json(
        tmp_path / "regime_diagnostics.json",
        posterior_probs=np.array([[0.8, 0.2], [0.1, 0.9]]),
    )

    payload = json.loads((tmp_path / "regime_diagnostics.json").read_text(encoding="utf-8"))
    assert payload["switch_count"] == 1
