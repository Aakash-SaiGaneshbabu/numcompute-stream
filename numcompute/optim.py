"""Optimisation utilities for NumCompute."""

from __future__ import annotations

import numpy as np


def numerical_gradient(f, x, eps=1e-6):
    """
    Compute a finite-difference gradient of scalar function f at x.
    """
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array.")

    grad = np.zeros_like(x, dtype=float)
    for i in range(x.size):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += eps
        x2[i] -= eps
        grad[i] = (f(x1) - f(x2)) / (2.0 * eps)
    return grad


def gradient_descent(f, grad_f, x0, lr=0.01, max_iter=1000, tol=1e-8):
    """
    Minimise f using gradient descent.
    Returns x_opt, history.
    """
    x = np.asarray(x0, dtype=float).copy()
    history = [x.copy()]

    for _ in range(max_iter):
        g = np.asarray(grad_f(x), dtype=float)
        if np.linalg.norm(g) < tol:
            break
        x = x - lr * g
        history.append(x.copy())
    return x, np.array(history)


def stochastic_gradient_descent(grad_f, x0, data, lr=0.01, epochs=10, shuffle=True, tol=1e-8, random_state=None):
    """
    Generic SGD loop over data rows.
    grad_f(x, sample) -> gradient vector
    """
    rng = np.random.default_rng(random_state)
    x = np.asarray(x0, dtype=float).copy()
    data = np.asarray(data)
    history = [x.copy()]

    for _ in range(epochs):
        indices = np.arange(data.shape[0])
        if shuffle:
            rng.shuffle(indices)

        for idx in indices:
            g = np.asarray(grad_f(x, data[idx]), dtype=float)
            if np.linalg.norm(g) < tol:
                continue
            x = x - lr * g
        history.append(x.copy())

    return x, np.array(history)