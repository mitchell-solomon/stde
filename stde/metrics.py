import numpy as np


def convergence_std(loss_curve: np.ndarray, window: int = 10) -> float:
    """Measure convergence of a noisy loss curve.

    The metric is defined as the standard deviation of the difference between
    the noisy curve and its smoothed version obtained via a moving average of
    the specified window size.

    Args:
        loss_curve: Sequence of loss values.
        window: Size of the moving average window used for smoothing.

    Returns:
        The standard deviation of the difference between the noisy and smoothed
        curves.
    """
    if window < 1:
        raise ValueError("window must be at least 1")
    loss_curve = np.asarray(loss_curve)
    kernel = np.ones(window) / window
    smooth_curve = np.convolve(loss_curve, kernel, mode="same")
    return float(np.std(loss_curve - smooth_curve))
