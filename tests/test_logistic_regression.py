# -*- coding: utf-8 -*-
"""
Tests for Logistic Regression algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LogisticRegression"))

from LogisticRegression import (
    loadtxtAndcsv_data,
    mapFeature,
    costFunction,
    gradient,
    sigmoid,
    predict,
)


class TestLoadData:
    def test_load_data1_success(self, logistic_regression_data1_path):
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        assert data is not None
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2

    def test_load_data2_success(self, logistic_regression_data2_path):
        data = loadtxtAndcsv_data(logistic_regression_data2_path, ",", np.float64)
        assert data is not None
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2

    def test_load_data_shape(self, logistic_regression_data1_path):
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        assert data.shape[0] > 0
        assert data.shape[1] == 3


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

    def test_sigmoid_large_positive(self):
        z = np.array([100])
        h = sigmoid(z)
        assert_almost_equal(h[0], 1.0, decimal=5)

    def test_sigmoid_large_negative(self):
        z = np.array([-100])
        h = sigmoid(z)
        assert_almost_equal(h[0], 0.0, decimal=5)

    def test_sigmoid_array(self):
        z = np.array([-10, 0, 10])
        h = sigmoid(z)
        assert h.shape == z.shape
        assert h[0] < 0.5
        assert_almost_equal(h[1], 0.5, decimal=5)
        assert h[2] > 0.5

    def test_sigmoid_range(self):
        z = np.linspace(-10, 10, 100)
        h = sigmoid(z)
        assert np.all(h > 0) and np.all(h < 1)


class TestMapFeature:
    def test_map_feature_output_shape(self):
        X1 = np.array([1, 2, 3])
        X2 = np.array([1, 2, 3])
        out = mapFeature(X1, X2)
        assert out.shape[0] == len(X1)
        assert out.shape[1] > 2

    def test_map_feature_first_column_ones(self):
        X1 = np.array([1, 2, 3])
        X2 = np.array([1, 2, 3])
        out = mapFeature(X1, X2)
        assert_array_almost_equal(out[:, 0], np.ones(len(X1)))

    def test_map_feature_contains_original(self):
        X1 = np.array([1.0, 2.0, 3.0])
        X2 = np.array([4.0, 5.0, 6.0])
        out = mapFeature(X1, X2)
        assert np.any(np.isclose(out[:, 1], X1))
        assert np.any(np.isclose(out[:, 2], X2))


class TestCostFunction:
    def test_cost_function_zero_theta(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        J = costFunction(theta, X_with_bias, y, Lambda)
        assert J >= 0
        assert isinstance(J, (float, np.ndarray))

    def test_cost_function_initial_cost(self):
        X = np.array([[1, 1, 0], [1, 1, 1], [1, 1, 2]])
        y = np.array([0, 1, 1])
        theta = np.zeros(3)
        Lambda = 0
        J = costFunction(theta, X, y, Lambda)
        assert_almost_equal(float(J), 0.693147, decimal=3)

    def test_cost_function_with_regularization(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 1.0
        J_reg = costFunction(theta, X_with_bias, y, Lambda)
        Lambda = 0.0
        J_no_reg = costFunction(theta, X_with_bias, y, Lambda)
        assert J_reg >= J_no_reg

    def test_cost_function_perfect_fit(self):
        X = np.array([[1, 1, 0], [1, 1, 10], [1, 1, -10]])
        y = np.array([0, 1, 0])
        theta = np.array([0, 0, 1])
        Lambda = 0
        J = costFunction(theta, X, y, Lambda)
        assert J >= 0


class TestGradient:
    def test_gradient_shape(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        grad = gradient(theta, X_with_bias, y, Lambda)
        assert grad.shape == theta.shape

    def test_gradient_zero_theta(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        grad = gradient(theta, X_with_bias, y, Lambda)
        assert isinstance(grad, np.ndarray)

    def test_gradient_with_regularization(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.ones(X_with_bias.shape[1])
        Lambda = 1.0
        grad = gradient(theta, X_with_bias, y, Lambda)
        assert grad is not None


class TestPredict:
    def test_predict_output_shape(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        p = predict(X_with_bias, theta)
        assert p.shape[0] == m

    def test_predict_binary_values(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        p = predict(X_with_bias, theta)
        assert np.all((p == 0) | (p == 1))

    def test_predict_perfect_separation(self):
        X = np.array([[1, 1, 10], [1, 1, -10], [1, 1, 20], [1, 1, -20]])
        y = np.array([1, 0, 1, 0])
        theta = np.array([0, 0, 1])
        p = predict(X, theta)
        assert_array_almost_equal(p.flatten(), y)


class TestLogisticRegressionIntegration:
    def test_full_pipeline(self, logistic_regression_data1_path):
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        J = costFunction(theta, X_with_bias, y, Lambda)
        grad = gradient(theta, X_with_bias, y, Lambda)
        assert J >= 0
        assert grad.shape == theta.shape

    def test_optimization_with_scipy(self, logistic_regression_data1_path):
        from scipy import optimize
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        result = optimize.fmin_bfgs(
            costFunction, theta, fprime=gradient, args=(X_with_bias, y, Lambda), maxiter=100
        )
        assert result is not None
        assert len(result) == X_with_bias.shape[1]

    def test_accuracy_on_data(self, logistic_regression_data1_path):
        from scipy import optimize
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        result = optimize.fmin_bfgs(
            costFunction, theta, fprime=gradient, args=(X_with_bias, y, Lambda), maxiter=100
        )
        p = predict(X_with_bias, result)
        accuracy = np.mean(p.flatten() == y)
        assert accuracy > 0.5


class TestLogisticRegressionSklearnComparison:
    def test_sklearn_comparison(self, logistic_regression_data1_path):
        from scipy import optimize
        from sklearn.linear_model import LogisticRegression
        data = loadtxtAndcsv_data(logistic_regression_data1_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 0.1
        result = optimize.fmin_bfgs(
            costFunction, theta, fprime=gradient, args=(X_with_bias, y, Lambda), maxiter=100
        )
        p_custom = predict(X_with_bias, result)
        model = LogisticRegression(C=1.0 / Lambda, solver="lbfgs", max_iter=100)
        model.fit(X, y)
        p_sklearn = model.predict(X)
        custom_accuracy = np.mean(p_custom.flatten() == y)
        sklearn_accuracy = np.mean(p_sklearn == y)
        assert abs(custom_accuracy - sklearn_accuracy) < 0.2


class TestEdgeCases:
    def test_single_sample(self):
        X = np.array([[1, 1.0, 2.0]])
        y = np.array([1])
        theta = np.zeros(3)
        Lambda = 0.1
        J = costFunction(theta, X, y, Lambda)
        grad = gradient(theta, X, y, Lambda)
        assert J >= 0
        assert grad.shape == theta.shape

    def test_all_same_class(self):
        X = np.array([[1, 1.0], [1, 2.0], [1, 3.0]])
        y = np.array([1, 1, 1])
        theta = np.zeros(2)
        Lambda = 0.1
        J = costFunction(theta, X, y, Lambda)
        assert J >= 0

    def test_large_lambda(self, simple_classification_data):
        X, y = simple_classification_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros(X_with_bias.shape[1])
        Lambda = 100.0
        J = costFunction(theta, X_with_bias, y, Lambda)
        grad = gradient(theta, X_with_bias, y, Lambda)
        assert J >= 0
        assert grad is not None


class TestOneVsAll:
    def test_onevsall_data_load(self, logistic_regression_digits_path):
        import scipy.io as spio
        data = spio.loadmat(logistic_regression_digits_path)
        assert "X" in data
        assert "y" in data
        assert data["X"].shape[0] > 0
        assert data["y"].shape[0] > 0

    def test_onevsall_data_shape(self, logistic_regression_digits_path):
        import scipy.io as spio
        data = spio.loadmat(logistic_regression_digits_path)
        X = data["X"]
        y = data["y"]
        assert X.shape[1] == 400
        assert len(np.unique(y)) <= 10
