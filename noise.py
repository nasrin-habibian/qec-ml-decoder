"""Noise generation for the three-qubit bit-flip code."""

import numpy as np


def sample_bit_flip_errors(
    n_samples: int,
    error_probability: float,
    random_state: int | None = None,
) -> np.ndarray:
    """
    Generate independent X/bit-flip errors for three physical qubits.

    Returns
    -------
    errors : ndarray of shape (n_samples, 3)
        Each entry is 0 (no flip) or 1 (bit flip).
    """
    rng = np.random.default_rng(random_state)
    return rng.binomial(
        n=1,
        p=error_probability,
        size=(n_samples, 3),
    ).astype(int)
