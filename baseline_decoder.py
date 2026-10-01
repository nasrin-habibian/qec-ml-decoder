"""Rule-based syndrome decoder."""

import numpy as np


def decode_syndrome(syndromes: np.ndarray) -> np.ndarray:
    """
    Decode syndromes using the standard lookup for single bit-flip errors.

    Syndrome mapping:
        00 -> no error
        10 -> qubit 1
        11 -> qubit 2
        01 -> qubit 3

    Returns class labels:
        0 = no error
        1 = qubit 1
        2 = qubit 2
        3 = qubit 3
    """
    syndromes = np.asarray(syndromes, dtype=int)

    if syndromes.ndim != 2 or syndromes.shape[1] != 2:
        raise ValueError("syndromes must have shape (n_samples, 2)")

    predictions = np.zeros(len(syndromes), dtype=int)

    mapping = {
        (0, 0): 0,
        (1, 0): 1,
        (1, 1): 2,
        (0, 1): 3,
    }

    for i, syndrome in enumerate(syndromes):
        predictions[i] = mapping[tuple(syndrome)]

    return predictions
