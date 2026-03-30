# -*- coding: utf-8 -*-
"""
Unit tests for Neural Network module.
Tests cover: data loading, sigmoid functions, cost function, gradient,
forward propagation, backpropagation, weight initialization, and prediction.
"""
import os
import sys
import numpy as np
import pytest
from scipy import optimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'NeuralNetwok'))

from conftest import (
    NEURAL_NETWORK_DIR, assert_array_almost_equal_custom, assert_shape
)


def sigmoid(z):
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def sigmoidGradient(z):
    g = sigmoid(z) * (1 - sigmoid(z))
    return g


def randInitializeWeights(L_in, L_out):
    W = np.zeros((L_out, 1 + L_in))
    epsilon_init = (6.0 / (L_out + L_in)) ** 0.5
    W = np.random.rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init
    return W


def debugInitializeWeights(fan_in, fan_out):
    W = np.zeros((fan_out, fan_in + 1))
    x = np.arange(1, fan_out * (fan_in + 1) + 1)
    W = np.sin(x).reshape(W.shape) / 10
    return W


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels, hidden_layer_size + 1)
    m = X.shape[0]
    class_y = np.zeros((m, num_labels))
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]
    term = np.dot(np.transpose(np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1)))), np.vstack((Theta1_x.reshape(-1, 1), Theta2_x.reshape(-1, 1))))
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)
    J = -(np.dot(np.transpose(class_y.reshape(-1, 1)), np.log(h.reshape(-1, 1))) + np.dot(np.transpose(1 - class_y.reshape(-1, 1)), np.log(1 - h.reshape(-1, 1))) - Lambda * term / 2) / m
    return np.ravel(J)


def nnGradient(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda):
    length = nn_params.shape[0]
    Theta1 = nn_params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1).copy()
    Theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels, hidden_layer_size + 1).copy()
    m = X.shape[0]
    class_y = np.zeros((m, num_labels))
    for i in range(num_labels):
        class_y[:, i] = np.int32(y == i).reshape(1, -1)
    Theta1_colCount = Theta1.shape[1]
    Theta1_x = Theta1[:, 1:Theta1_colCount]
    Theta2_colCount = Theta2.shape[1]
    Theta2_x = Theta2[:, 1:Theta2_colCount]
    Theta1_grad = np.zeros((Theta1.shape))
    Theta2_grad = np.zeros((Theta2.shape))
    a1 = np.hstack((np.ones((m, 1)), X))
    z2 = np.dot(a1, np.transpose(Theta1))
    a2 = sigmoid(z2)
    a2 = np.hstack((np.ones((m, 1)), a2))
    z3 = np.dot(a2, np.transpose(Theta2))
    h = sigmoid(z3)
    delta3 = np.zeros((m, num_labels))
    delta2 = np.zeros((m, hidden_layer_size))
    for i in range(m):
        delta3[i, :] = h[i, :] - class_y[i, :]
        Theta2_grad = Theta2_grad + np.dot(np.transpose(delta3[i, :].reshape(1, -1)), a2[i, :].reshape(1, -1))
        delta2[i, :] = np.dot(delta3[i, :].reshape(1, -1), Theta2_x) * sigmoidGradient(z2[i, :])
        Theta1_grad = Theta1_grad + np.dot(np.transpose(delta2[i, :].reshape(1, -1)), a1[i, :].reshape(1, -1))
    Theta1[:, 0] = 0
    Theta2[:, 0] = 0
    grad = (np.vstack((Theta1_grad.reshape(-1, 1), Theta2_grad.reshape(-1, 1))) + Lambda * np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))) / m
    return np.ravel(grad)


def predict(Theta1, Theta2, X):
    m = X.shape[0]
    num_labels = Theta2.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    h1 = sigmoid(np.dot(X, np.transpose(Theta1)))
    h1 = np.hstack((np.ones((m, 1)), h1))
    h2 = sigmoid(np.dot(h1, np.transpose(Theta2)))
    p = np.array(np.where(h2[0, :] == np.max(h2, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(h2[i, :] == np.max(h2, axis=1)[i]))
        p = np.vstack((p, t))
    return p


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_digits_data(self, digits_data):
        """Test that digits data can be loaded"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        assert digits_data is not None, "Data should be loaded"

    def test_digits_data_shape(self, digits_data):
        """Test that digits data has correct shape"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        y = digits_data['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 400, "Each digit should have 400 features (20x20)"

    def test_digits_data_values(self, digits_data):
        """Test that digits data values are valid"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        y = digits_data['y']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isinf(X)), "X should not contain infinite values"


class TestSigmoidFunctions:
    """Tests for sigmoid and sigmoid gradient functions"""

    def test_sigmoid_output_range(self):
        """Test that sigmoid output is always between 0 and 1"""
        z = np.array([-100, -10, -1, 0, 1, 10, 100])
        h = sigmoid(z)
        assert np.all(h > 0) and np.all(h < 1), "Sigmoid output should be between 0 and 1"

    def test_sigmoid_zero(self):
        """Test sigmoid at zero"""
        z = np.array([0])
        h = sigmoid(z)
        np.testing.assert_almost_equal(h[0], 0.5, decimal=10)

    def test_sigmoid_gradient_range(self):
        """Test that sigmoid gradient is always between 0 and 0.25"""
        z = np.array([-100, -10, -1, 0, 1, 10, 100])
        g = sigmoidGradient(z)
        assert np.all(g >= 0) and np.all(g <= 0.25), "Sigmoid gradient should be between 0 and 0.25"

    def test_sigmoid_gradient_zero(self):
        """Test sigmoid gradient at zero"""
        z = np.array([0])
        g = sigmoidGradient(z)
        np.testing.assert_almost_equal(g[0], 0.25, decimal=10)

    def test_sigmoid_gradient_large_values(self):
        """Test sigmoid gradient approaches 0 for large values"""
        z = np.array([100, -100])
        g = sigmoidGradient(z)
        np.testing.assert_array_almost_equal(g, [0, 0], decimal=5)


class TestWeightInitialization:
    """Tests for weight initialization"""

    def test_rand_initialize_weights_shape(self):
        """Test that random initialization produces correct shape"""
        L_in = 3
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        assert W.shape == (L_out, L_in + 1), f"Weight shape should be ({L_out}, {L_in + 1})"

    def test_rand_initialize_weights_range(self):
        """Test that random initialization produces values in expected range"""
        L_in = 10
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        epsilon = (6.0 / (L_out + L_in)) ** 0.5
        assert np.all(W >= -epsilon) and np.all(W <= epsilon), "Weights should be within epsilon range"

    def test_rand_initialize_weights_not_zero(self):
        """Test that random initialization does not produce all zeros"""
        L_in = 3
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        assert not np.all(W == 0), "Weights should not all be zero"

    def test_debug_initialize_weights_shape(self):
        """Test debug initialization produces correct shape"""
        fan_in = 3
        fan_out = 5
        W = debugInitializeWeights(fan_in, fan_out)
        assert W.shape == (fan_out, fan_in + 1), f"Weight shape should be ({fan_out}, {fan_in + 1})"

    def test_debug_initialize_weights_deterministic(self):
        """Test that debug initialization is deterministic"""
        fan_in = 3
        fan_out = 5
        W1 = debugInitializeWeights(fan_in, fan_out)
        W2 = debugInitializeWeights(fan_in, fan_out)
        np.testing.assert_array_equal(W1, W2, "Debug initialization should be deterministic")


class TestCostFunction:
    """Tests for neural network cost function"""

    def test_cost_function_shape(self, small_neural_network_params):
        """Test that cost function returns scalar"""
        params = small_neural_network_params
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params.flatten(), params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 0)
        assert np.isscalar(J) or J.shape == (), "Cost should be scalar"

    def test_cost_function_positive(self, small_neural_network_params):
        """Test that cost function returns positive value"""
        params = small_neural_network_params
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        J = nnCostFunction(nn_params.flatten(), params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 0)
        assert J >= 0, "Cost should be non-negative"

    def test_cost_function_with_regularization(self, small_neural_network_params):
        """Test that regularization increases cost"""
        params = small_neural_network_params
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        J_no_reg = nnCostFunction(nn_params.flatten(), params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 0)
        J_with_reg = nnCostFunction(nn_params.flatten(), params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 1)
        assert J_with_reg > J_no_reg, "Regularization should increase cost"


class TestGradient:
    """Tests for neural network gradient computation"""

    def test_gradient_shape(self, small_neural_network_params):
        """Test that gradient has correct shape"""
        params = small_neural_network_params
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        grad = nnGradient(nn_params.flatten(), params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 0)
        assert grad.shape == nn_params.flatten().shape, "Gradient should have same shape as parameters"

    def test_gradient_numerical_check(self, small_neural_network_params):
        """Test gradient using numerical gradient checking"""
        params = small_neural_network_params
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1))).flatten()
        Lambda = 0.1
        grad = nnGradient(nn_params, params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, Lambda)
        epsilon = 1e-5
        num_grad = np.zeros(nn_params.shape)
        for i in range(len(nn_params)):
            params_plus = nn_params.copy()
            params_plus[i] += epsilon
            params_minus = nn_params.copy()
            params_minus[i] -= epsilon
            loss1 = nnCostFunction(params_minus, params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, Lambda)
            loss2 = nnCostFunction(params_plus, params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, Lambda)
            num_grad[i] = (loss2 - loss1) / (2 * epsilon)
        np.testing.assert_array_almost_equal(grad, num_grad, decimal=4, err_msg="Gradient check failed")


class TestPrediction:
    """Tests for neural network prediction"""

    def test_predict_shape(self):
        """Test that prediction has correct shape"""
        np.random.seed(42)
        m = 10
        input_size = 5
        hidden_size = 3
        num_labels = 3
        X = np.random.randn(m, input_size)
        Theta1 = np.random.randn(hidden_size, input_size + 1)
        Theta2 = np.random.randn(num_labels, hidden_size + 1)
        p = predict(Theta1, Theta2, X)
        assert p.shape[0] == m, "Prediction should have same number of samples"

    def test_predict_valid_classes(self):
        """Test that predictions are valid class indices"""
        np.random.seed(42)
        m = 10
        input_size = 5
        hidden_size = 3
        num_labels = 3
        X = np.random.randn(m, input_size)
        Theta1 = np.random.randn(hidden_size, input_size + 1)
        Theta2 = np.random.randn(num_labels, hidden_size + 1)
        p = predict(Theta1, Theta2, X)
        assert np.all((p >= 0) & (p < num_labels)), "Predictions should be valid class indices"


class TestNeuralNetworkIntegration:
    """Integration tests for neural network"""

    def test_neural_network_training_small(self, small_neural_network_params):
        """Test neural network training on small network"""
        params = small_neural_network_params
        initial_Theta1 = randInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = randInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        Lambda = 0.1
        result = optimize.fmin_cg(
            nnCostFunction, initial_nn_params.flatten(), fprime=nnGradient,
            args=(params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, Lambda),
            maxiter=10
        )
        assert result is not None, "Optimization should complete"

    def test_neural_network_end_to_end(self, digits_data):
        """Test complete neural network pipeline on digits data"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        y = digits_data['y']
        m, n = X.shape
        input_layer_size = 400
        hidden_layer_size = 25
        num_labels = 10
        Lambda = 1
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        result = optimize.fmin_cg(
            nnCostFunction, initial_nn_params.flatten(), fprime=nnGradient,
            args=(input_layer_size, hidden_layer_size, num_labels, X, y, Lambda),
            maxiter=10
        )
        length = result.shape[0]
        Theta1 = result[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
        Theta2 = result[hidden_layer_size * (input_layer_size + 1):length].reshape(num_labels, hidden_layer_size + 1)
        p = predict(Theta1, Theta2, X)
        accuracy = np.mean(np.float64(p == y.reshape(-1, 1))) * 100
        assert accuracy > 5, "Accuracy should be better than random guessing (10% for 10 classes)"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample"""
        np.random.seed(42)
        m = 1
        input_size = 5
        hidden_size = 3
        num_labels = 3
        X = np.random.randn(m, input_size)
        y = np.array([[0]])
        Theta1 = np.random.randn(hidden_size, input_size + 1)
        Theta2 = np.random.randn(num_labels, hidden_size + 1)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1))).flatten()
        J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        assert not np.isnan(J), "Cost should not be NaN"

    def test_single_hidden_unit(self):
        """Test with single hidden unit"""
        np.random.seed(42)
        m = 10
        input_size = 3
        hidden_size = 1
        num_labels = 2
        X = np.random.randn(m, input_size)
        y = np.random.randint(0, 2, (m, 1))
        Theta1 = np.random.randn(hidden_size, input_size + 1)
        Theta2 = np.random.randn(num_labels, hidden_size + 1)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1))).flatten()
        J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        assert not np.isnan(J), "Cost should not be NaN"

    def test_high_regularization(self):
        """Test with high regularization"""
        params = {
            'input_layer_size': 3,
            'hidden_layer_size': 5,
            'num_labels': 3,
            'm': 5
        }
        initial_Theta1 = debugInitializeWeights(params['input_layer_size'], params['hidden_layer_size'])
        initial_Theta2 = debugInitializeWeights(params['hidden_layer_size'], params['num_labels'])
        X = debugInitializeWeights(params['input_layer_size'] - 1, params['m'])
        y = np.transpose(np.mod(np.arange(1, params['m'] + 1), params['num_labels']))
        y = y.reshape(-1, 1)
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1))).flatten()
        J_low = nnCostFunction(nn_params, params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 0.01)
        J_high = nnCostFunction(nn_params, params['input_layer_size'], params['hidden_layer_size'], params['num_labels'], X, y, 100)
        assert J_high > J_low, "Higher regularization should increase cost"
