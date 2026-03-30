# -*- coding: utf-8 -*-
"""
Unit tests for Linear Regression module.
Tests cover: data loading, cost function, gradient descent, feature normalization,
prediction, and comparison with scikit-learn.
"""
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'LinearRegression'))

from conftest import (
    LINEAR_REGRESSION_DIR, assert_array_almost_equal_custom, assert_shape
)


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def featureNormaliza(X):
    X_norm = np.array(X)
    mu = np.zeros((1, X.shape[1]))
    sigma = np.zeros((1, X.shape[1]))
    mu = np.mean(X_norm, 0)
    sigma = np.std(X_norm, 0)
    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:, i] - mu[i]) / sigma[i]
    return X_norm, mu, sigma


def computerCost(X, y, theta):
    m = len(y)
    J = (np.transpose(X * theta - y)) * (X * theta - y) / (2 * m)
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    m = len(y)
    n = len(theta)
    temp = np.matrix(np.zeros((n, num_iters)))
    J_history = np.zeros((num_iters, 1))
    for i in range(num_iters):
        h = np.dot(X, theta)
        temp[:, i] = theta - ((alpha / m) * (np.dot(np.transpose(X), h - y)))
        theta = temp[:, i]
        J_history[i] = computerCost(X, y, theta)
    return theta, J_history


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_txt_data_exists(self, linear_regression_data):
        """Test that linear regression data file can be loaded"""
        assert linear_regression_data is not None, "Data file should exist"

    def test_load_txt_data_shape(self, linear_regression_data):
        """Test that loaded data has correct shape"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 2, "X should have 2 features"
        assert len(y.shape) == 1 or y.shape[1] == 1, "y should be 1D or column vector"

    def test_load_txt_data_values(self, linear_regression_data):
        """Test that loaded data values are valid"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isnan(y)), "y should not contain NaN values"
        assert not np.any(np.isinf(X)), "X should not contain infinite values"
        assert not np.any(np.isinf(y)), "y should not contain infinite values"


class TestFeatureNormalization:
    """Tests for feature normalization functionality"""

    def test_feature_normalization_output_shape(self, synthetic_regression_data):
        """Test that normalization preserves shape"""
        X = synthetic_regression_data['X']
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape, "Normalized X should have same shape as input"

    def test_feature_normalization_mean_zero(self, synthetic_regression_data):
        """Test that normalized features have approximately zero mean"""
        X = synthetic_regression_data['X']
        X_norm, mu, sigma = featureNormaliza(X)
        np.testing.assert_array_almost_equal(
            np.mean(X_norm, axis=0), np.zeros(X.shape[1]), decimal=10
        )

    def test_feature_normalization_std_one(self, synthetic_regression_data):
        """Test that normalized features have approximately unit std"""
        X = synthetic_regression_data['X']
        X_norm, mu, sigma = featureNormaliza(X)
        np.testing.assert_array_almost_equal(
            np.std(X_norm, axis=0), np.ones(X.shape[1]), decimal=10
        )

    def test_feature_normalization_mu_sigma_shape(self, synthetic_regression_data):
        """Test that mu and sigma have correct shapes"""
        X = synthetic_regression_data['X']
        X_norm, mu, sigma = featureNormaliza(X)
        assert mu.shape[1] == X.shape[1], "mu should have same number of features"
        assert sigma.shape[1] == X.shape[1], "sigma should have same number of features"

    def test_feature_normalization_single_feature(self):
        """Test normalization with single feature"""
        X = np.array([[1], [2], [3], [4], [5]])
        X_norm, mu, sigma = featureNormaliza(X)
        np.testing.assert_almost_equal(np.mean(X_norm), 0, decimal=10)
        np.testing.assert_almost_equal(np.std(X_norm), 1, decimal=10)


class TestCostFunction:
    """Tests for cost function computation"""

    def test_cost_function_zero_theta(self, synthetic_regression_data):
        """Test cost function with zero theta"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        J = computerCost(X_matrix, y_matrix, theta_matrix)
        assert J.shape == (1, 1), "Cost should be a scalar (1x1 matrix)"
        assert J[0, 0] >= 0, "Cost should be non-negative"

    def test_cost_function_perfect_fit(self):
        """Test cost function with perfect fit (should be near zero)"""
        np.random.seed(42)
        m = 10
        X = np.random.randn(m, 2)
        true_theta = np.array([[1.5], [-2.0], [1.0]])
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        y = np.dot(X_with_bias, true_theta)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(true_theta)
        J = computerCost(X_matrix, y_matrix, theta_matrix)
        np.testing.assert_almost_equal(J[0, 0], 0, decimal=10)

    def test_cost_function_shape(self, synthetic_regression_data):
        """Test that cost function returns correct shape"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        theta = np.ones((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        J = computerCost(X_matrix, y_matrix, theta_matrix)
        assert J.shape == (1, 1), "Cost should be 1x1 matrix"


class TestGradientDescent:
    """Tests for gradient descent algorithm"""

    def test_gradient_descent_convergence(self, synthetic_regression_data):
        """Test that gradient descent converges (cost decreases)"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=100
        )
        assert J_history[-1, 0] < J_history[0, 0], "Cost should decrease"

    def test_gradient_descent_theta_shape(self, synthetic_regression_data):
        """Test that theta has correct shape after gradient descent"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=50
        )
        assert theta_final.shape == theta.shape, "Theta shape should remain constant"

    def test_gradient_descent_j_history_shape(self, synthetic_regression_data):
        """Test that J_history has correct shape"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        num_iters = 50
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=num_iters
        )
        assert J_history.shape == (num_iters, 1), f"J_history should have shape ({num_iters}, 1)"


class TestLinearRegressionIntegration:
    """Integration tests for linear regression"""

    def test_linear_regression_end_to_end(self, linear_regression_data):
        """Test complete linear regression pipeline"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=400
        )
        assert J_history[-1, 0] < J_history[0, 0], "Final cost should be lower than initial"
        assert theta_final.shape[0] == X.shape[1] + 1, "Theta should have bias term"

    def test_linear_regression_prediction(self, linear_regression_data):
        """Test prediction with trained model"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=400
        )
        predict = np.array([1650, 3])
        norm_predict = (predict - mu) / sigma
        final_predict = np.hstack((np.ones((1)), norm_predict))
        result = np.dot(final_predict, theta_final)
        assert result.shape == (1, 1) or result.shape == (1,), "Prediction should be scalar"


class TestSklearnComparison:
    """Tests comparing our implementation with scikit-learn"""

    def test_sklearn_linear_regression(self, linear_regression_data):
        """Test scikit-learn linear regression on same data"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        from sklearn import linear_model
        from sklearn.preprocessing import StandardScaler
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        scaler = StandardScaler()
        scaler.fit(X)
        x_train = scaler.transform(X)
        model = linear_model.LinearRegression()
        model.fit(x_train, y)
        assert model.coef_ is not None, "Model should have coefficients"
        assert model.intercept_ is not None, "Model should have intercept"

    def test_comparison_with_sklearn(self, linear_regression_data):
        """Compare our implementation with scikit-learn"""
        if linear_regression_data is None:
            pytest.skip("Data file not available")
        from sklearn import linear_model
        from sklearn.preprocessing import StandardScaler
        X = linear_regression_data['X']
        y = linear_regression_data['y']
        m = len(y)
        scaler = StandardScaler()
        x_train = scaler.fit_transform(X)
        model = linear_model.LinearRegression()
        model.fit(x_train, y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y_col = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y_col)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.1, num_iters=1000
        )
        theta_final = np.array(theta_final).flatten()
        sklearn_theta = np.array([model.intercept_] + list(model.coef_))
        np.testing.assert_array_almost_equal(
            theta_final, sklearn_theta, decimal=1,
            err_msg="Our implementation should match sklearn approximately"
        )


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample (edge case)"""
        X = np.array([[1.0, 2.0]])
        y = np.array([5.0])
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape

    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(50, 1)
        y = 2 * X.flatten() + 1 + np.random.randn(50) * 0.1
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y_col = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y_col)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=100
        )
        assert J_history[-1, 0] < J_history[0, 0], "Cost should decrease"

    def test_constant_features(self):
        """Test with constant feature values"""
        X = np.ones((10, 2))
        y = np.arange(10, dtype=np.float64)
        with pytest.raises((ValueError, ZeroDivisionError, RuntimeWarning)):
            X_norm, mu, sigma = featureNormaliza(X)

    def test_large_learning_rate(self, synthetic_regression_data):
        """Test with large learning rate (may cause divergence)"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y_col = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y_col)
        theta_matrix = np.matrix(theta)
        theta_final, J_history = gradientDescent(
            X_matrix, y_matrix, theta_matrix, alpha=10.0, num_iters=10
        )
        assert not np.any(np.isnan(J_history)), "J_history should not contain NaN"

    def test_zero_iterations(self, synthetic_regression_data):
        """Test with zero iterations"""
        X = synthetic_regression_data['X']
        y = synthetic_regression_data['y']
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_with_bias.shape[1], 1))
        y_col = y.reshape(-1, 1)
        X_matrix = np.matrix(X_with_bias)
        y_matrix = np.matrix(y_col)
        theta_matrix = np.matrix(theta)
        with pytest.raises(IndexError):
            theta_final, J_history = gradientDescent(
                X_matrix, y_matrix, theta_matrix, alpha=0.01, num_iters=0
            )
