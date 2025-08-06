import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stde.metrics import convergence_std


def test_convergence_std(tmp_path):
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 10.0, 100)
    trend = np.exp(-x / 3.0)
    losses = trend + 0.1 * rng.standard_normal(x.shape)

    metric = convergence_std(losses, window=5)

    expected_smooth = np.convolve(losses, np.ones(5) / 5, mode="same")
    expected_metric = np.std(losses - expected_smooth)
    assert np.isclose(metric, expected_metric)

    plt.figure()
    plt.plot(x, losses, label="noisy")
    plt.plot(x, expected_smooth, label="smoothed")
    plt.legend()
    out_path = tmp_path / "loss_plot.png"
    plt.savefig(out_path)
    assert out_path.exists()
