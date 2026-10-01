# Machine Learning for Quantum Error Correction

A small-scale computational study of machine-learning-based decoding for a three-qubit bit-flip quantum error-correcting code.

## Research question

Can a machine-learning model learn to infer bit-flip errors from noisy syndrome measurements and achieve reliable decoding performance?

## Project overview

This project implements a simple three-qubit repetition code for correcting single-qubit bit-flip errors. It compares:

1. A rule-based syndrome decoder (baseline)
2. A machine-learning decoder based on logistic regression

The experiment evaluates both decoders under different physical bit-flip noise rates.

The project is intentionally a small-scale study. It is designed to demonstrate the computational ideas behind QEC decoding rather than to model a full fault-tolerant quantum computer.

## Pipeline

Physical noise
→ error pattern
→ syndrome measurement
→ decoder
→ predicted error
→ logical error evaluation

## Code structure

```text
qec-ml-decoder/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── noise.py
│   ├── syndrome.py
│   ├── baseline_decoder.py
│   ├── ml_decoder.py
│   └── evaluation.py
├── notebooks/
│   └── qec_ml_experiment.ipynb
├── figures/
└── results/
```

## Methods

### Three-qubit bit-flip code

The code represents one logical bit using three physical qubits:

```text
|0_L> = |000>
|1_L> = |111>
```

The code is designed to correct a single bit-flip (X) error.

The two parity checks are:

```text
s1 = q1 XOR q2
s2 = q2 XOR q3
```

The syndrome identifies which physical qubit has flipped for the single-error cases.

### Baseline decoder

The baseline decoder uses the standard syndrome-to-error lookup:

```text
00 -> no correction
10 -> qubit 1
11 -> qubit 2
01 -> qubit 3
```

### Machine-learning decoder

A logistic-regression classifier receives the two syndrome bits and predicts the most likely single-qubit error class.

This provides a simple data-driven baseline for comparing learned decoding with a deterministic syndrome decoder.

## Experiments

The experiment generates random bit-flip errors at several physical error probabilities:

- 1%
- 2%
- 5%
- 10%
- 15%

For each noise level, the project reports decoder accuracy and logical-error behaviour.

## Limitations

This is a simplified educational/research prototype:

- It models a three-qubit bit-flip code.
- It uses classical syndrome data rather than a full quantum hardware simulation.
- It considers bit-flip errors rather than a general quantum noise model.
- The machine-learning model is deliberately simple.

The purpose is to study the basic relationship between QEC syndromes, decoding, and machine learning before moving to larger codes and more realistic noise models.

## Reproducibility

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the experiment:

```bash
python -m src.evaluation
```

The experiment prints a performance table and saves results to:

```text
results/decoder_results.csv
```

## Technologies

Python | NumPy | scikit-learn | pandas | Matplotlib

## Research context

The project explores the intersection of machine learning, error-correcting codes, and quantum information, with an emphasis on data-driven decoding under noisy conditions.
