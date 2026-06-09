import numpy as np
import pytest

from numcompute.optim import numerical_gradient, gradient_descent, stochastic_gradient_descent


def test_numerical_gradient_quadratic():
    def f(x):
        return x[0] ** 2 + 3 * x[1] ** 2

    x = np.array([2.0, -1.0])
    grad = numerical_gradient(f, x)
    expected = np.array([4.0, -6.0])
    assert np.allclose(grad, expected, atol=1e-4)


def test_numerical_gradient_invalid_dimension():
    def f(x):
        return np.sum(x**2)

    with pytest.raises(ValueError):
        numerical_gradient(f, np.array([[1.0, 2.0]]))


def test_gradient_descent_simple_quadratic():
    def f(x):
        return (x[0] - 3) ** 2 + (x[1] + 2) ** 2

    def grad_f(x):
        return np.array([2 * (x[0] - 3), 2 * (x[1] + 2)])

    x0 = np.array([0.0, 0.0])
    x_opt, history = gradient_descent(f, grad_f, x0, lr=0.1, max_iter=500, tol=1e-10)

    assert np.allclose(x_opt, np.array([3.0, -2.0]), atol=1e-4)
    assert history.shape[1] == 2


def test_gradient_descent_stops_when_gradient_small():
    def f(x):
        return x[0] ** 2

    def grad_f(x):
        return np.array([0.0])

    x0 = np.array([5.0])
    x_opt, history = gradient_descent(f, grad_f, x0)

    assert np.array_equal(x_opt, x0)
    assert history.shape == (1, 1)


def test_stochastic_gradient_descent_simple():
    def grad_f(x, sample):
        target = sample[0]
        return np.array([2 * (x[0] - target)])

    data = np.array([[1.0], [2.0], [3.0], [4.0]])
    x0 = np.array([0.0])

    x_opt, history = stochastic_gradient_descent(
        grad_f,
        x0,
        data,
        lr=0.05,
        epochs=20,
        shuffle=False,
        random_state=42,
    )

    assert x_opt.shape == (1,)
    assert history.shape[1] == 1


def test_stochastic_gradient_descent_invalid_input():
    def grad_f(x, sample):
        return np.array([1.0])

    x0 = np.array([0.0])
    data = np.array([1.0, 2.0, 3.0])

    x_opt, history = stochastic_gradient_descent(grad_f, x0, data, epochs=2)
    assert x_opt.shape == (1,)
    assert history.shape[1] == 1
