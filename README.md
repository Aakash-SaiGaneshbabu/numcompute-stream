# NumCompute-Stream

NumCompute-Stream is a NumPy-based streaming machine learning framework developed as an extension of the NumCompute numerical computing toolkit.

The project provides:

* Streaming preprocessing
* Decision Tree Classifier
* Random Forest and Bagging Ensembles
* Online Statistics
* Streaming Metrics
* Incremental Training Utilities
* Benchmarking Tools
* Data Visualisation
* NumPy-only Implementation

## Project Structure

```text
numcompute-individual/
│
├── numcompute/
├── tests/
├── demo/
├── benchmark/
├── README.md
├── pyproject.toml
└── .gitignore
```

## Installation

```bash
pip install -e .
```

## Running Tests

```bash
pytest -v
```

## Running Benchmarks

```bash
python benchmark/run_benchmarks.py
```

## Running Demo

```bash
python demo/demo_stream.py
```

## Features

### Streaming Learning

* Chunk-based training
* Online updates
* Incremental preprocessing
* Running evaluation metrics

### Machine Learning

* Decision Trees
* Bagging
* Random Forests

### Visualisation

* Metric tracking
* Confusion matrices
* ROC curves
* Model comparisons
* Memory monitoring

### Benchmarking

* Loop vs vectorised comparisons
* Tree vs ensemble comparisons
* Streaming performance evaluation

## Dependencies

* NumPy
* matplotlib
* pytest (testing)
