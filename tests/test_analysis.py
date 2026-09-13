import numpy as np

from atomic.numerics.analysis import count_sign_changes


def test_counts_true_sign_changes():
    x = np.linspace(0, 2 * np.pi, 1000)
    assert count_sign_changes(np.sin(x + 0.1)) == 2

def test_ignores_noise_below_floor():
    y = np.ones(100)
    y[50:] = -1.0
    y[10] = -1e-12
    assert count_sign_changes(y) == 1
