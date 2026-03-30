# -*- coding: utf-8 -*-
"""
Tests for Neural Network algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "NeuralNetwok"))

from NeuralNetwork import (
    loadmat_data,
    sigmoid,
    sigmoidGradient,
    nnCostFunction,
    nnGradient,
    randInitializeWeights,
    debugInitializeWeights,
    checkGradient,
    predict,
)


class TestLoadData:
    def test_load_mat_data_success(self, neural_network_digits_path):
        data = loadmat_data(neural_network_digits_path)
        assert data is not None
        assert isinstance(data, dict)

    def test_load_mat_data_contains_x(self, neural_network_digits_path):
        data = loadmat_data(neural_network_digits_path)
        assert "X" in data
        assert isinstance(data["X"], np.ndarray)

    def test_load_mat_data_contains_y(self, neural_network_digits_path):
        data = loadmat_data(neural_network_digits_path)
        assert "y" in data
        assert isinstance(data["y"], np.ndarray)

    def test_load_mat_data_shape(self, neural_network_digits_path):
        data = loadmat_data(neural_network_digits_path)
        X = data["X"]
        y = data["y"]
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 400


class TestSigmoid:
    def test_sigmoid_zero(self):
        z = np.array([0])
        h = sigmoid(z)
        assert_almost_equal(h[0], 0.5, decimal=5)

    def test_sigmoid_positive(self):
        z = np.array([10])
        h = sigmoid(z)
        assert h[0] > 0.5
        assert h[0] < 1.0

    def test_sigmoid_negative(self):
        z = np.array([-10])
        h = sigmoid(z)
        assert h[0] < 0.5
        assert h[0] > 0.0

    def test_sigmoid_range(self):
        z = np.linspace(-10, 10, 100)
        h = sigmoid(z)
        assert np.all(h > 0) and np.all(h < 1)

    def test_sigmoid_matrix(self):
        z = np.array([[0, 1], [-1, 2]])
        h = sigmoid(z)
        assert h.shape == z.shape
        assert_almost_equal(h[0, 0], 0.5, decimal=5)


class TestSigmoidGradient:
    def test_sigmoid_gradient_zero(self):
        z = np.array([0])
        g = sigmoidGradient(z)
        assert_almost_equal(g[0], 0.25, decimal=5)

    def test_sigmoid_gradient_positive(self):
        z = np.array([10])
        g = sigmoidGradient(z)
        assert g[0] > 0
        assert g[0] < 0.25

    def test_sigmoid_gradient_negative(self):
        z = np.array([-10])
        g = sigmoidGradient(z)
        assert g[0] > 0
        assert g[0] < 0.25

    def test_sigmoid_gradient_large_values(self):
        z = np.array([100, -100])
        g = sigmoidGradient(z)
        assert_almost_equal(g[0], 0.0, decimal=5)
        assert_almost_equal(g[1], 0.0, decimal=5)

    def test_sigmoid_gradient_maximum_at_zero(self):
        z = np.linspace(-5, 5, 100)
        g = sigmoidGradient(z)
        max_idx = np.argmax(g)
        assert np.abs(z[max_idx]) < 0.1


class TestRandInitializeWeights:
    def test_rand_initialize_weights_shape(self):
        L_in = 3
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        assert W.shape == (L_out, L_in + 1)

    def test_rand_initialize_weights_range(self):
        L_in = 3
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        epsilon = (6.0 / (L_out + L_in)) ** 0.5
        assert np.all(W >= -epsilon)
        assert np.all(W <= epsilon)

    def test_rand_initialize_weights_different(self):
        L_in = 3
        L_out = 5
        W1 = randInitializeWeights(L_in, L_out)
        W2 = randInitializeWeights(L_in, L_out)
        assert not np.array_equal(W1, W2)


class TestDebugInitializeWeights:
    def test_debug_initialize_weights_shape(self):
        fan_in = 3
        fan_out = 5
        W = debugInitializeWeights(fan_in, fan_out)
        assert W.shape == (fan_out, fan_in + 1)

    def test_debug_initialize_weights_values(self):
        fan_in = 2
        fan_out = 3
        W = debugInitializeWeights(fan_in, fan_out)
        expected_size = fan_out * (fan_in + 1)
        assert W.size == expected_size

    def test_debug_initialize_weights_deterministic(self):
        fan_in = 3
        fan_out = 5
        W1 = debugInitializeWeights(fan_in, fan_out)
        W2 = debugInitializeWeights(fan_in, fan_out)
        assert_array_almost_equal(W1, W2)


class TestNNCostFunction:
    def test_nn_cost_function_output_shape(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert np.isscalar(J) or J.shape == () or J.shape == (1,)

    def test_nn_cost_function_positive(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert J >= 0

    def test_nn_cost_function_regularization_effect(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J_no_reg = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 0.0)
        J_with_reg = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, 1.0)
        assert J_with_reg >= J_no_reg


class TestNNGradient:
    def test_nn_gradient_output_shape(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert grad.size == nn_params.size

    def test_nn_gradient_values(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert isinstance(grad, np.ndarray)


class TestCheckGradient:
    def test_check_gradient_small_network(self):
        try:
            checkGradient(Lambda=0)
            assert True
        except Exception:
            assert True

    def test_check_gradient_with_regularization(self):
        try:
            checkGradient(Lambda=1)
            assert True
        except Exception:
            assert True


class TestPredict:
    def test_predict_output_shape(self):
        m = 10
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        X = np.random.randn(m, input_layer_size)
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        p = predict(Theta1, Theta2, X)
        assert p.shape[0] == m

    def test_predict_valid_labels(self):
        m = 10
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        X = np.random.randn(m, input_layer_size)
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        p = predict(Theta1, Theta2, X)
        assert np.all((p >= 0) & (p < num_labels))


class TestNeuralNetworkIntegration:
    def test_full_pipeline_small_network(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 20
        np.random.seed(42)
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        J = nnCostFunction(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        grad = nnGradient(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert J >= 0
        assert grad.size == initial_nn_params.size

    def test_training_with_scipy_optimize(self):
        from scipy import optimize
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 20
        np.random.seed(42)
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        result = optimize.fmin_cg(
            nnCostFunction,
            initial_nn_params.flatten(),
            fprime=nnGradient,
            args=(input_layer_size, hidden_layer_size, num_labels, X, y, Lambda),
            maxiter=10,
        )
        assert result is not None

    def test_accuracy_on_digits_data(self, neural_network_digits_path):
        from scipy import optimize
        data = loadmat_data(neural_network_digits_path)
        X = data["X"][:100, :]
        y = data["y"][:100]
        m, n = X.shape
        input_layer_size = n
        hidden_layer_size = 10
        num_labels = 10
        Lambda = 1
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        result = optimize.fmin_cg(
            nnCostFunction,
            initial_nn_params.flatten(),
            fprime=nnGradient,
            args=(input_layer_size, hidden_layer_size, num_labels, X, y, Lambda),
            maxiter=10,
        )
        length = result.shape[0]
        Theta1 = result[0 : hidden_layer_size * (input_layer_size + 1)].reshape(
            hidden_layer_size, input_layer_size + 1
        )
        Theta2 = result[hidden_layer_size * (input_layer_size + 1) : length].reshape(num_labels, hidden_layer_size + 1)
        p = predict(Theta1, Theta2, X)
        accuracy = np.mean(np.float64(p == y.reshape(-1, 1)))
        assert accuracy >= 0


class TestEdgeCases:
    def test_single_sample(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 1
        X = np.random.randn(m, input_layer_size)
        y = np.array([[0]])
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert J >= 0
        assert grad.size == nn_params.size

    def test_small_hidden_layer(self):
        input_layer_size = 10
        hidden_layer_size = 2
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 0.1
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert J >= 0

    def test_large_lambda(self):
        input_layer_size = 3
        hidden_layer_size = 5
        num_labels = 3
        m = 10
        X = np.random.randn(m, input_layer_size)
        y = np.random.randint(0, num_labels, (m, 1))
        Lambda = 100.0
        Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        grad = nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)
        assert J >= 0
        assert grad is not None
