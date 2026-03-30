# -*- coding: utf-8 -*-
"""
Unit tests for Logistic Regression module.
Tests cover: data loading, sigmoid function, cost function, gradient,
mapFeature, prediction, OneVsAll, and comparison with scikit-learn.
"""
import os
import sys
import numpy as np
import pytest
from scipy import optimize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'LogisticRegression'))

from conftest import (
    LOGISTIC_REGRESSION_DIR, assert_array_almost_equal_custom, assert_shape
)


def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def sigmoid(z):
    h = np.zeros((len(z), 1))
    h = 1.0 / (1.0 + np.exp(-z))
    return h


def mapFeature(X1, X2, degree=2):
    out = np.ones((X1.shape[0], 1))
    for i in np.arange(1, degree + 1):
        for j in range(i + 1):
            temp = X1 ** (i - j) * (X2 ** j)
            out = np.hstack((out, temp.reshape(-1, 1)))
    return out


def costFunction(initial_theta, X, y, inital_lambda):
    m = len(y)
    J = 0
    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0
    temp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y), np.log(1 - h)) + temp * inital_lambda / 2) / m
    return J


def gradient(initial_theta, X, y, inital_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))
    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0
    grad = np.dot(np.transpose(X), h - y) / m + inital_lambda / m * theta1
    return grad


def predict(X, theta):
    m = X.shape[0]
    p = np.zeros((m, 1))
    p = sigmoid(np.dot(X, theta))
    for i in range(m):
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_data1_exists(self, logistic_regression_data1):
        """Test that logistic regression data1 file can be loaded"""
        assert logistic_regression_data1 is not None, "Data1 file should exist"

    def test_load_data2_exists(self, logistic_regression_data2):
        """Test that logistic regression data2 file can be loaded"""
        assert logistic_regression_data2 is not None, "Data2 file should exist"

    def test_load_data1_shape(self, logistic_regression_data1):
        """Test that loaded data1 has correct shape"""
        if logistic_regression_data1 is None:
            pytest.skip("Data1 file not available")
        X = logistic_regression_data1['X']
        y = logistic_regression_data1['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 2, "X should have 2 features"

    def test_load_data2_shape(self, logistic_regression_data2):
        """Test that loaded data2 has correct shape"""
        if logistic_regression_data2 is None:
            pytest.skip("Data2 file not available")
        X = logistic_regression_data2['X']
        y = logistic_regression_data2['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"

    def test_load_digits_data(self, digits_data):
        """Test that digits data can be loaded"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        y = digits_data['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 400, "Each digit should have 400 features (20x20)"


class TestSigmoidFunction:
    """Tests for sigmoid function"""

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

    def test_sigmoid_symmetry(self):
        """Test sigmoid symmetry: sigmoid(-z) = 1 - sigmoid(z)"""
        z = np.array([1, 2, 3, 4, 5])
        h_pos = sigmoid(z)
        h_neg = sigmoid(-z)
        np.testing.assert_array_almost_equal(h_neg, 1 - h_pos, decimal=10)

    def test_sigmoid_large_positive(self):
        """Test sigmoid approaches 1 for large positive values"""
        z = np.array([100])
        h = sigmoid(z)
        np.testing.assert_almost_equal(h[0], 1, decimal=5)

    def test_sigmoid_large_negative(self):
        """Test sigmoid approaches 0 for large negative values"""
        z = np.array([-100])
        h = sigmoid(z)
        np.testing.assert_almost_equal(h[0], 0, decimal=5)

    def test_sigmoid_shape(self):
        """Test that sigmoid preserves input shape"""
        z = np.array([[1, 2], [3, 4], [5, 6]])
        h = sigmoid(z.flatten())
        assert len(h) == 6, "Sigmoid should flatten and return 1D array"


class TestMapFeature:
    """Tests for polynomial feature mapping"""

    def test_map_feature_shape(self):
        """Test that mapFeature produces correct output shape"""
        X1 = np.array([1, 2, 3])
        X2 = np.array([4, 5, 6])
        out = mapFeature(X1, X2, degree=2)
        assert out.shape[0] == 3, "Output should have same number of samples"
        assert out.shape[1] == 6, "For degree=2, should have 6 features (1, x1, x2, x1^2, x1*x2, x2^2)"

    def test_map_feature_degree1(self):
        """Test mapFeature with degree=1"""
        X1 = np.array([1, 2])
        X2 = np.array([3, 4])
        out = mapFeature(X1, X2, degree=1)
        assert out.shape[1] == 3, "For degree=1, should have 3 features (1, x1, x2)"

    def test_map_feature_degree3(self):
        """Test mapFeature with degree=3"""
        X1 = np.array([1, 2])
        X2 = np.array([3, 4])
        out = mapFeature(X1, X2, degree=3)
        assert out.shape[1] == 10, "For degree=3, should have 10 features"

    def test_map_feature_values(self):
        """Test that mapFeature produces correct polynomial values"""
        X1 = np.array([2])
        X2 = np.array([3])
        out = mapFeature(X1, X2, degree=2)
        expected = np.array([[1, 2, 3, 4, 6, 9]])
        np.testing.assert_array_almost_equal(out, expected, decimal=10)


class TestCostFunction:
    """Tests for cost function computation"""

    def test_cost_function_initial_theta(self):
        """Test cost function with initial theta (zeros)"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.zeros(3)
        J = costFunction(theta, X, y, 0)
        np.testing.assert_almost_equal(J, np.log(2), decimal=5)

    def test_cost_function_perfect_prediction(self):
        """Test cost function with perfect prediction"""
        X = np.array([[1, 0], [1, 10]])
        y = np.array([0, 1])
        theta = np.array([-10, 1])
        J = costFunction(theta, X, y, 0)
        assert J < 0.01, "Cost should be very small for perfect prediction"

    def test_cost_function_with_regularization(self):
        """Test cost function with regularization"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.array([1, 2, 3])
        J_no_reg = costFunction(theta, X, y, 0)
        J_with_reg = costFunction(theta, X, y, 1)
        assert J_with_reg > J_no_reg, "Regularization should increase cost"

    def test_cost_function_shape(self):
        """Test that cost function returns scalar"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.zeros(3)
        J = costFunction(theta, X, y, 0)
        assert np.isscalar(J) or J.shape == (), "Cost should be scalar"


class TestGradient:
    """Tests for gradient computation"""

    def test_gradient_shape(self):
        """Test that gradient has correct shape"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.zeros(3)
        grad = gradient(theta, X, y, 0)
        assert grad.shape == theta.shape, "Gradient should have same shape as theta"

    def test_gradient_numerical_check(self):
        """Test gradient using numerical gradient checking"""
        np.random.seed(42)
        m = 5
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.array([0, 1, 0, 1, 0]).astype(np.float64)
        theta = np.random.randn(3)
        grad = gradient(theta, X, y, 0.1)
        epsilon = 1e-5
        num_grad = np.zeros(3)
        for i in range(3):
            theta_plus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus = theta.copy()
            theta_minus[i] -= epsilon
            num_grad[i] = (costFunction(theta_plus, X, y, 0.1) - costFunction(theta_minus, X, y, 0.1)) / (2 * epsilon)
        np.testing.assert_array_almost_equal(grad, num_grad, decimal=4)

    def test_gradient_zero_theta(self):
        """Test gradient with zero theta"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.zeros(3)
        grad = gradient(theta, X, y, 0)
        assert grad is not None, "Gradient should be computed"


class TestPrediction:
    """Tests for prediction functionality"""

    def test_predict_binary_output(self):
        """Test that prediction produces binary output"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        theta = np.array([0, 1, -1])
        p = predict(X, theta)
        assert np.all((p == 0) | (p == 1)), "Predictions should be binary (0 or 1)"

    def test_predict_shape(self):
        """Test that prediction has correct shape"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        theta = np.array([0, 1, -1])
        p = predict(X, theta)
        assert p.shape[0] == m, "Prediction should have same number of samples"

    def test_predict_threshold(self):
        """Test prediction threshold at 0.5"""
        X = np.array([[1, 0], [1, 0], [1, 0]])
        theta = np.array([0, 10])
        p = predict(X, theta)
        np.testing.assert_array_equal(p.flatten(), [0, 0, 0])


class TestLogisticRegressionIntegration:
    """Integration tests for logistic regression"""

    def test_logistic_regression_end_to_end(self, logistic_regression_data1):
        """Test complete logistic regression pipeline"""
        if logistic_regression_data1 is None:
            pytest.skip("Data1 file not available")
        X = logistic_regression_data1['X']
        y = logistic_regression_data1['y']
        m = len(y)
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        initial_theta = np.zeros(X_with_bias.shape[1])
        initial_lambda = 0.1
        result = optimize.fmin_bfgs(
            costFunction, initial_theta, fprime=gradient,
            args=(X_with_bias, y, initial_lambda), maxiter=100
        )
        p = predict(X_with_bias, result)
        accuracy = np.mean(p.flatten() == y) * 100
        assert accuracy > 50, "Accuracy should be better than random guessing"

    def test_logistic_regression_with_polynomial_features(self, logistic_regression_data2):
        """Test logistic regression with polynomial features"""
        if logistic_regression_data2 is None:
            pytest.skip("Data2 file not available")
        X = logistic_regression_data2['X']
        y = logistic_regression_data2['y']
        X_poly = mapFeature(X[:, 0], X[:, 1], degree=2)
        initial_theta = np.zeros(X_poly.shape[1])
        initial_lambda = 0.1
        result = optimize.fmin_bfgs(
            costFunction, initial_theta, fprime=gradient,
            args=(X_poly, y, initial_lambda), maxiter=100
        )
        p = predict(X_poly, result)
        accuracy = np.mean(p.flatten() == y) * 100
        assert accuracy > 50, "Accuracy should be better than random guessing"


class TestOneVsAll:
    """Tests for OneVsAll multi-class classification"""

    def test_one_vs_all_training(self, digits_data):
        """Test OneVsAll training on digits data"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        y = digits_data['y']
        m, n = X.shape
        num_labels = 10
        all_theta = np.zeros((n + 1, num_labels))
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        class_y = np.zeros((m, num_labels))
        for i in range(num_labels):
            class_y[:, i] = np.int32(y == i).reshape(1, -1)
        initial_theta = np.zeros((n + 1, 1))
        Lambda = 0.1
        for i in range(num_labels):
            result = optimize.fmin_bfgs(
                costFunction, initial_theta, fprime=gradient,
                args=(X_with_bias, class_y[:, i], Lambda), maxiter=10
            )
            all_theta[:, i] = result
        assert all_theta.shape == (n + 1, num_labels), "all_theta should have correct shape"

    def test_one_vs_all_prediction_shape(self, digits_data):
        """Test OneVsAll prediction output shape"""
        if digits_data is None:
            pytest.skip("Digits data file not available")
        X = digits_data['X']
        m = X.shape[0]
        num_labels = 10
        all_theta = np.random.randn(num_labels, X.shape[1] + 1)
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        h = sigmoid(np.dot(X_with_bias, np.transpose(all_theta)))
        assert h.shape == (m, num_labels), "Prediction probabilities should have correct shape"


class TestSklearnComparison:
    """Tests comparing our implementation with scikit-learn"""

    def test_sklearn_logistic_regression(self, logistic_regression_data1):
        """Test scikit-learn logistic regression on same data"""
        if logistic_regression_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = logistic_regression_data1['X']
        y = logistic_regression_data1['y']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LogisticRegression()
        model.fit(X_scaled, y)
        accuracy = model.score(X_scaled, y)
        assert accuracy > 0.8, "Sklearn accuracy should be reasonable"

    def test_comparison_with_sklearn(self, logistic_regression_data1):
        """Compare our implementation with scikit-learn"""
        if logistic_regression_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        X = logistic_regression_data1['X']
        y = logistic_regression_data1['y']
        m = len(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_with_bias = np.hstack((np.ones((m, 1)), X_scaled))
        initial_theta = np.zeros(X_with_bias.shape[1])
        result = optimize.fmin_bfgs(
            costFunction, initial_theta, fprime=gradient,
            args=(X_with_bias, y, 0), maxiter=100
        )
        p_ours = predict(X_with_bias, result)
        model = LogisticRegression(fit_intercept=False)
        model.fit(X_with_bias, y)
        p_sklearn = model.predict(X_with_bias)
        our_accuracy = np.mean(p_ours.flatten() == y)
        sklearn_accuracy = np.mean(p_sklearn == y)
        assert abs(our_accuracy - sklearn_accuracy) < 0.1, "Our accuracy should be close to sklearn"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[1, 2, 3]])
        y = np.array([1])
        theta = np.array([0, 0, 0, 0])
        J = costFunction(theta, X, y, 0)
        assert not np.isnan(J), "Cost should not be NaN"

    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        m = 50
        X = np.random.randn(m, 1)
        y = (X[:, 0] > 0).astype(np.float64)
        X_with_bias = np.hstack((np.ones((m, 1)), X))
        initial_theta = np.zeros(2)
        result = optimize.fmin_bfgs(
            costFunction, initial_theta, fprime=gradient,
            args=(X_with_bias, y, 0), maxiter=50
        )
        p = predict(X_with_bias, result)
        accuracy = np.mean(p.flatten() == y)
        assert accuracy > 0.9, "Accuracy should be high for linearly separable data"

    def test_all_same_class(self):
        """Test with all samples in same class"""
        X = np.random.randn(10, 2)
        y = np.ones(10)
        X_with_bias = np.hstack((np.ones((10, 1)), X))
        initial_theta = np.zeros(3)
        J = costFunction(initial_theta, X_with_bias, y, 0)
        assert not np.isnan(J), "Cost should not be NaN"

    def test_high_regularization(self):
        """Test with high regularization"""
        np.random.seed(42)
        m = 10
        X = np.hstack((np.ones((m, 1)), np.random.randn(m, 2)))
        y = np.random.randint(0, 2, m).astype(np.float64)
        theta = np.array([1, 2, 3])
        J_low = costFunction(theta, X, y, 0.01)
        J_high = costFunction(theta, X, y, 100)
        assert J_high > J_low, "Higher regularization should increase cost"
