"""Run the QEC decoder comparison experiment."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from .baseline_decoder import decode_syndrome
from .ml_decoder import train_ml_decoder
from .noise import sample_bit_flip_errors
from .syndrome import compute_syndromes, error_class


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10, 0.15]


def evaluate_decoders(
    n_train: int = 20_000,
    n_test: int = 20_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compare the baseline and ML decoders across noise levels.

    The ML model is trained separately for each noise level so that
    the experiment measures decoding under the same distribution
    used to generate the test set.
    """
    rows = []

    for i, p in enumerate(NOISE_LEVELS):
        train_errors = sample_bit_flip_errors(
            n_train, p, random_state=random_state + i
        )
        test_errors = sample_bit_flip_errors(
            n_test, p, random_state=random_state + 100 + i
        )

        X_train = compute_syndromes(train_errors)
        X_test = compute_syndromes(test_errors)

        y_train = error_class(train_errors)
        y_test = error_class(test_errors)

        # Keep only classes the repetition-code decoder is designed
        # to identify: no error and single-qubit errors.
        train_mask = y_train <= 3
        test_mask = y_test <= 3

        X_train = X_train[train_mask]
        y_train = y_train[train_mask]

        X_test = X_test[test_mask]
        y_test = y_test[test_mask]

        baseline_pred = decode_syndrome(X_test)

        model = train_ml_decoder(X_train, y_train, random_state=random_state)
        ml_pred = model.predict(X_test)

        baseline_accuracy = accuracy_score(y_test, baseline_pred)
        ml_accuracy = accuracy_score(y_test, ml_pred)

        rows.append(
            {
                "noise_probability": p,
                "baseline_accuracy": baseline_accuracy,
                "ml_accuracy": ml_accuracy,
                "test_samples": len(y_test),
            }
        )

    return pd.DataFrame(rows)


def save_results(results: pd.DataFrame) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    csv_path = RESULTS_DIR / "decoder_results.csv"
    results.to_csv(csv_path, index=False)

    plt.figure(figsize=(7, 5))
    plt.plot(
        results["noise_probability"] * 100,
        results["baseline_accuracy"],
        marker="o",
        label="Baseline decoder",
    )
    plt.plot(
        results["noise_probability"] * 100,
        results["ml_accuracy"],
        marker="o",
        label="ML decoder",
    )
    plt.xlabel("Physical bit-flip probability (%)")
    plt.ylabel("Decoder accuracy")
    plt.title("QEC Decoder Performance")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    figure_path = FIGURES_DIR / "decoder_performance.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()

    print(f"Results saved to: {csv_path}")
    print(f"Figure saved to: {figure_path}")


def main() -> None:
    results = evaluate_decoders()

    print("\nDecoder comparison:")
    print(results.to_string(index=False))

    save_results(results)


if __name__ == "__main__":
    main()
