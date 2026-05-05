import json

import numpy as np

from aaf.eval.artifacts import write_mixture_diagnostics_json
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
