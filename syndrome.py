"""Syndrome generation for the three-qubit bit-flip code."""

import numpy as np


def compute_syndromes(errors: np.ndarray) -> np.ndarray:
    """
    Compute the two parity-check syndrome bits.

    For errors [e1, e2, e3]:

        s1 = e1 XOR e2
        s2 = e2 XOR e3

    Returns
    -------
    syndrome : ndarray of shape (n_samples, 2)
    """
    errors = np.asarray(errors, dtype=int)

    if errors.ndim != 2 or errors.shape[1] != 3:
        raise ValueError("errors must have shape (n_samples, 3)")

    s1 = np.bitwise_xor(errors[:, 0], errors[:, 1])
    s2 = np.bitwise_xor(errors[:, 1], errors[:, 2])

    return np.column_stack([s1, s2])


def error_class(errors: np.ndarray) -> np.ndarray:
    """
    Convert an error pattern into a simple class label.

    Classes:
        0 = no error
        1 = qubit 1
        2 = qubit 2
        3 = qubit 3
        4 = multiple flips

    The multiple-flip class is retained for analysis, but the
    three-qubit repetition code is only guaranteed to correct
    single bit-flip errors.
    """
    errors = np.asarray(errors, dtype=int)
    weights = errors.sum(axis=1)

    labels = np.full(len(errors), 4, dtype=int)
    labels[weights == 0] = 0

    for qubit in range(3):
        mask = (weights == 1) & (errors[:, qubit] == 1)
        labels[mask] = qubit + 1

    return labels
