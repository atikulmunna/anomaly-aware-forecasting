import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from aaf.models.mdn_lstm import MDNLSTMConfig, MDNLSTMForecaster  # noqa: E402
from aaf.models.mixture import mixture_nll  # noqa: E402


def test_mdn_lstm_forward_shapes() -> None:
    model = MDNLSTMForecaster(
        MDNLSTMConfig(
            input_size=2,
            output_size=2,
            hidden_size=8,
            num_layers=1,
            horizon=3,
            n_components=4,
        )
    )
    history = torch.zeros(5, 7, 2)

    params = model(history)

    assert tuple(params.logits.shape) == (5, 7, 3, 4)
    assert tuple(params.means.shape) == (5, 7, 3, 4, 2)
    assert tuple(params.raw_stds.shape) == (5, 7, 3, 4, 2)
    assert torch.all(params.stds > 0.0)


def test_forecast_last_keeps_single_time_dimension() -> None:
    model = MDNLSTMForecaster(
        MDNLSTMConfig(input_size=1, output_size=1, hidden_size=4, num_layers=1, horizon=2)
    )

    params = model.forecast_last(torch.zeros(3, 6, 1))

    assert tuple(params.logits.shape) == (3, 1, 2, 3)
    assert tuple(params.means.shape) == (3, 1, 2, 3, 1)


def test_mdn_lstm_loss_backpropagates() -> None:
    model = MDNLSTMForecaster(
        MDNLSTMConfig(input_size=1, output_size=1, hidden_size=6, num_layers=1, horizon=2)
    )
    history = torch.randn(4, 5, 1)
    target = torch.randn(4, 5, 2, 1)

    loss = mixture_nll(target, model(history))
    loss.backward()

    assert torch.isfinite(loss)
    assert model.mean_head.weight.grad is not None
    assert torch.isfinite(model.mean_head.weight.grad).all()


def test_mdn_lstm_rejects_wrong_input_size() -> None:
    model = MDNLSTMForecaster(MDNLSTMConfig(input_size=2, output_size=1))

    with pytest.raises(ValueError, match="input_size"):
        model(torch.zeros(1, 3, 1))


def test_mdn_lstm_config_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="horizon"):
        MDNLSTMConfig(input_size=1, output_size=1, horizon=0).validate()
