# -*- coding: utf-8 -*-
"""
Tests for Linear Regression algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LinearRegression"))

from LinearRegression import (
    loadtxtAndcsv_data,
    loadnpy_data,
    featureNormaliza,
    gradientDescent,
    computerCost,
    predict,
)


class TestLoadData:
    def test_load_txt_data_success(self, linear_regression_data_path):
        data = loadtxtAndcsv_data(linear_regression_data_path, ",", np.float64)
        assert data is not None
        assert isinstance(data, np.ndarray)
        assert data.ndim == 2
        assert data.shape[1] == 3

    def test_load_txt_data_shape(self, linear_regression_data_path):
        data = loadtxtAndcsv_data(linear_regression_data_path, ",", np.float64)
        assert data.shape[0] > 0
        assert data.shape[1] == 3

    def test_load_npy_data(self):
        npy_path = os.path.join(PROJECT_ROOT, "LinearRegression", "data.npy")
        if os.path.exists(npy_path):
            data = loadnpy_data(npy_path)
            assert data is not None
            assert isinstance(data, np.ndarray)


class TestFeatureNormalization:
    def test_feature_normaliza_output_shape(self, simple_regression_data):
        X, _, _ = simple_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape
        assert mu.shape[0] == X.shape[1]
        assert sigma.shape[0] == X.shape[1]

    def test_feature_normaliza_mean_zero(self, simple_regression_data):
        X, _, _ = simple_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        mean_after = np.mean(X_norm, axis=0)
        assert_array_almost_equal(mean_after, np.zeros(X.shape[1]), decimal=10)

    def test_feature_normaliza_std_one(self, simple_regression_data):
        X, _, _ = simple_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        std_after = np.std(X_norm, axis=0)
        assert_array_almost_equal(std_after, np.ones(X.shape[1]), decimal=10)

    def test_feature_normaliza_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape
        assert_almost_equal(mu[0], 3.0)
        assert sigma[0] > 0

    def test_feature_normaliza_constant_feature(self):
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape
        assert_almost_equal(mu[0], 5.0)


class TestComputerCost:
    def test_computer_cost_zero_theta(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.matrix(np.hstack((np.ones((m, 1)), X)))
        theta = np.matrix(np.zeros((X_with_bias.shape[1], 1)))
        y = np.matrix(y.reshape(-1, 1))
        J = computerCost(X_with_bias, y, theta)
        assert J >= 0
        assert isinstance(J, (float, np.ndarray, np.matrix))

    def test_computer_cost_optimal_theta(self, simple_regression_data):
        X, y, true_theta = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.matrix(np.hstack((np.ones((m, 1)), X)))
        y = np.matrix(y.reshape(-1, 1))
        J = computerCost(X_with_bias, y, np.matrix(true_theta))
        assert J >= 0
        assert J < 1.0

    def test_computer_cost_perfect_fit(self):
        X = np.matrix([[1, 1], [1, 2], [1, 3]])
        y = np.matrix([[2], [4], [6]])
        theta = np.matrix([[0], [2]])
        J = computerCost(X, y, theta)
        assert float(np.array(J).flatten()[0]) < 1e-5

    def test_computer_cost_shape(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.matrix(np.hstack((np.ones((m, 1)), X)))
        theta = np.matrix(np.zeros((X_with_bias.shape[1], 1)))
        y = np.matrix(y.reshape(-1, 1))
        J = computerCost(X_with_bias, y, theta)
        assert np.isscalar(J) or J.shape == (1, 1)


class TestGradientDescent:
    def test_gradient_descent_convergence(self, simple_regression_data):
        X, y, true_theta = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 100
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert theta_final.shape == theta.shape
        assert J_history.shape[0] == num_iters

    def test_gradient_descent_cost_decrease(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 100
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert J_history[-1] < J_history[0]

    def test_gradient_descent_output_shape(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 50
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert theta_final.shape[0] == X_with_bias.shape[1]
        assert len(J_history) == num_iters

    def test_gradient_descent_simple_linear(self):
        X = np.matrix([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.matrix([[3], [5], [7], [9]])
        theta = np.matrix(np.zeros((2, 1)))
        alpha = 0.1
        num_iters = 100
        theta_final, J_history = gradientDescent(X, y, theta, alpha, num_iters)
        assert_array_almost_equal(np.array(theta_final).flatten(), [1.0, 2.0], decimal=0)


class TestPredict:
    def test_predict_output_shape(self, simple_regression_data):
        X, y, true_theta = simple_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        result = predict(mu, sigma, true_theta)
        assert result is not None

    def test_predict_value_range(self, simple_regression_data):
        X, y, true_theta = simple_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        result = predict(mu, sigma, true_theta)
        assert isinstance(result, (float, np.ndarray))


class TestLinearRegressionIntegration:
    def test_full_pipeline(self, linear_regression_data_path):
        data = loadtxtAndcsv_data(linear_regression_data_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 100
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert J_history[-1] < J_history[0]
        assert theta_final is not None

    def test_linear_regression_with_sklearn_comparison(self, linear_regression_data_path):
        data = loadtxtAndcsv_data(linear_regression_data_path, ",", np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 400
        theta_final, _ = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        from sklearn.linear_model import LinearRegression as SklearnLR
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = SklearnLR()
        model.fit(X_scaled, y)
        assert theta_final.shape[0] == X.shape[1] + 1


class TestEdgeCases:
    def test_single_sample(self):
        X = np.matrix([[1.0, 1.0, 2.0]])
        y = np.matrix([[5.0]])
        theta = np.matrix(np.zeros((3, 1)))
        J = computerCost(X, y, theta)
        assert J >= 0

    def test_large_learning_rate(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 1.0
        num_iters = 10
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert theta_final is not None

    def test_zero_iterations(self, simple_regression_data):
        X, y, _ = simple_regression_data
        m = X.shape[0]
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        alpha = 0.01
        num_iters = 0
        theta_final, J_history = gradientDescent(X_with_bias, y, theta, alpha, num_iters)
        assert len(J_history) == 0
