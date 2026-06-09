<<<<<<< HEAD
# Phase 2 Assignment 2.2
=======
>>>>>>>
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
<<<<<<< HEAD

# Phase 1 Assignment 2.1 - Base for Phase 2
# NumCompute-Stream

NumCompute-Stream is a modular, NumPy-only streaming machine learning framework built on top of the original NumCompute project. It supports chunk-wise learning, incremental preprocessing, online metrics, decision-tree learning, tree ensembles, and lightweight matplotlib visualisation.

## Key Features

- Streaming preprocessing with `partial_fit()`
  - `StandardScaler`
  - `MinMaxScaler`
  - `SimpleImputer`
  - `OneHotEncoder`
- Streaming statistics
  - running mean, variance, min, max
  - histogram accumulation
  - quantile tracking
- Streaming metrics
  - accuracy, precision, recall, F1
  - confusion matrix
  - MSE
  - ROC curve and AUC
- Tree-based learning
  - depth-limited `DecisionTreeClassifier`
  - `partial_fit()` support
  - Gini or entropy impurity
- Ensemble learning
  - `BaggingClassifier`
  - `RandomForestClassifier`
  - streaming adaptation across chunks
- Streaming orchestration
  - `StreamTrainer`
  - chunk-based logging
  - cumulative accuracy tracking
- Visualisation
  - metric-over-time plots
  - model comparison plots
  - predictions vs ground truth plots

## Requirements

- Python
- NumPy
- matplotlib
- pytest for testing

## Running the Demo

```bash
python demo/demo_stream.py
```

The demo creates a synthetic streaming classification problem, trains the model in chunks, logs metrics, and saves plots.

## Running Tests

```bash
python -m pytest -q
```

## Design Notes

- All core logic is implemented using NumPy only.
- The project uses incremental updates where possible to support streaming data.
- The same public API style is kept across modules for consistency and ease of testing.
- Visual outputs use matplotlib and can be shown inline or saved to file.

## Useful Imports

```python
import numcompute as nc
from numcompute.stream import StreamTrainer
from numcompute.ensemble import RandomForestClassifier
from numcompute.preprocessing import StandardScaler
from numcompute.visualise import plot_metric_over_time
```
=======
>>>>>>> dcd22fadc939c7f2405c4798c56ef9c3f61241ec
