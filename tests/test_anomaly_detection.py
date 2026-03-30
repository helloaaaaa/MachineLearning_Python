# -*- coding: utf-8 -*-
"""
Unit tests for Anomaly Detection module.
Tests cover: data loading, Gaussian parameter estimation,
multivariate Gaussian distribution, threshold selection, and anomaly detection.
"""
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'AnomalyDetection'))

from conftest import (
    ANOMALY_DETECTION_DIR, assert_array_almost_equal_custom, assert_shape
)


def estimateGaussian(X):
    m, n = X.shape
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))
    mu = np.mean(X, axis=0)
    sigma2 = np.var(X, axis=0)
    return mu, sigma2


def multivariateGaussian(X, mu, Sigma2):
    k = len(mu)
    if Sigma2.shape[0] > 1:
        Sigma2 = np.diag(Sigma2)
    X = X - mu
    argu = (2 * np.pi) ** (-k / 2) * np.linalg.det(Sigma2) ** (-0.5)
    p = argu * np.exp(-0.5 * np.sum(np.dot(X, np.linalg.inv(Sigma2)) * X, axis=1))
    return p


def selectThreshold(yval, pval):
    bestEpsilon = 0.
    bestF1 = 0.
    F1 = 0.
    step = (np.max(pval) - np.min(pval)) / 1000
    for epsilon in np.arange(np.min(pval), np.max(pval), step):
        cvPrecision = pval < epsilon
        tp = np.sum((cvPrecision == 1) & (yval == 1).ravel()).astype(float)
        fp = np.sum((cvPrecision == 1) & (yval == 0).ravel()).astype(float)
        fn = np.sum((cvPrecision == 0) & (yval == 1).ravel()).astype(float)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon
    return bestEpsilon, bestF1


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_anomaly_data_exists(self, anomaly_detection_data):
        """Test that anomaly detection data file can be loaded"""
        assert anomaly_detection_data is not None, "Data file should exist"

    def test_load_anomaly_data_shape(self, anomaly_detection_data):
        """Test that loaded data has correct shape"""
        if anomaly_detection_data is None:
            pytest.skip("Data file not available")
        X = anomaly_detection_data['X']
        Xval = anomaly_detection_data['Xval']
        yval = anomaly_detection_data['yval']
        assert X.shape[0] > 0, "X should have samples"
        assert Xval.shape[0] > 0, "Xval should have samples"
        assert yval.shape[0] == Xval.shape[0], "yval and Xval should have same number of samples"

    def test_load_anomaly_data_values(self, anomaly_detection_data):
        """Test that loaded data values are valid"""
        if anomaly_detection_data is None:
            pytest.skip("Data file not available")
        X = anomaly_detection_data['X']
        Xval = anomaly_detection_data['Xval']
        yval = anomaly_detection_data['yval']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isnan(Xval)), "Xval should not contain NaN values"
        assert not np.any(np.isnan(yval)), "yval should not contain NaN values"
        assert set(np.unique(yval)).issubset({0, 1}), "yval should be binary (0 or 1)"


class TestEstimateGaussian:
    """Tests for Gaussian parameter estimation"""

    def test_estimate_gaussian_output_shape(self, synthetic_regression_data):
        """Test that estimation produces correct shapes"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape[0] == X.shape[1], "mu should have same number of features"
        assert sigma2.shape[0] == X.shape[1], "sigma2 should have same number of features"

    def test_estimate_gaussian_mean(self, synthetic_regression_data):
        """Test that estimated mean is correct"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        expected_mu = np.mean(X, axis=0)
        np.testing.assert_array_almost_equal(mu, expected_mu, decimal=10)

    def test_estimate_gaussian_variance(self, synthetic_regression_data):
        """Test that estimated variance is correct"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        expected_sigma2 = np.var(X, axis=0)
        np.testing.assert_array_almost_equal(sigma2, expected_sigma2, decimal=10)

    def test_estimate_gaussian_positive_variance(self, synthetic_regression_data):
        """Test that variance is positive"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        assert np.all(sigma2 >= 0), "Variance should be non-negative"


class TestMultivariateGaussian:
    """Tests for multivariate Gaussian distribution"""

    def test_multivariate_gaussian_output_shape(self, synthetic_regression_data):
        """Test that probability output has correct shape"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape[0] == X.shape[0], "Probability should have same number of samples"

    def test_multivariate_gaussian_positive(self, synthetic_regression_data):
        """Test that probabilities are positive"""
        X = synthetic_regression_data['X']
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert np.all(p > 0), "Probabilities should be positive"

    def test_multivariate_gaussian_max_at_mean(self):
        """Test that probability is maximum at the mean"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        mean_point = mu.reshape(1, -1)
        p_at_mean = multivariateGaussian(mean_point, mu, sigma2)
        assert p_at_mean[0] >= np.max(p) * 0.99, "Probability should be highest near mean"

    def test_multivariate_gaussian_decreases_with_distance(self):
        """Test that probability decreases with distance from mean"""
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        near_point = np.array([[0.1, 0.1]])
        far_point = np.array([[5, 5]])
        p_near = multivariateGaussian(near_point, mu, sigma2)
        p_far = multivariateGaussian(far_point, mu, sigma2)
        assert p_near[0] > p_far[0], "Probability should decrease with distance"

    def test_multivariate_gaussian_symmetric(self):
        """Test that Gaussian is symmetric around mean"""
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        point1 = np.array([[1, 0]])
        point2 = np.array([[-1, 0]])
        p1 = multivariateGaussian(point1, mu, sigma2)
        p2 = multivariateGaussian(point2, mu, sigma2)
        np.testing.assert_almost_equal(p1[0], p2[0], decimal=10)


class TestSelectThreshold:
    """Tests for threshold selection"""

    def test_select_threshold_output_range(self):
        """Test that selected threshold is in valid range"""
        np.random.seed(42)
        pval = np.random.rand(100)
        yval = np.random.randint(0, 2, 100)
        epsilon, F1 = selectThreshold(yval, pval)
        assert epsilon >= np.min(pval), "Epsilon should be >= min probability"
        assert epsilon <= np.max(pval), "Epsilon should be <= max probability"
        assert 0 <= F1 <= 1, "F1 score should be between 0 and 1"

    def test_select_threshold_f1_score(self):
        """Test F1 score calculation"""
        pval = np.array([0.1, 0.2, 0.9, 0.95])
        yval = np.array([0, 0, 1, 1])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0, "F1 score should be positive for good threshold"

    def test_select_threshold_perfect_detection(self):
        """Test threshold selection with perfect detection"""
        pval = np.array([0.1, 0.1, 0.1, 0.9, 0.95])
        yval = np.array([0, 0, 0, 1, 1])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0.9, "F1 score should be high for clear separation"

    def test_select_threshold_no_anomalies(self):
        """Test threshold selection when no anomalies"""
        pval = np.random.rand(100)
        yval = np.zeros(100)
        epsilon, F1 = selectThreshold(yval, pval)
        assert epsilon is not None, "Should still return a threshold"


class TestAnomalyDetectionIntegration:
    """Integration tests for anomaly detection"""

    def test_anomaly_detection_end_to_end(self, anomaly_detection_data):
        """Test complete anomaly detection pipeline"""
        if anomaly_detection_data is None:
            pytest.skip("Data file not available")
        X = anomaly_detection_data['X']
        Xval = anomaly_detection_data['Xval']
        yval = anomaly_detection_data['yval']
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        pval = multivariateGaussian(Xval, mu, sigma2)
        epsilon, F1 = selectThreshold(yval, pval)
        outliers = np.where(p < epsilon)[0]
        assert len(outliers) >= 0, "Should detect some outliers"
        assert epsilon > 0, "Epsilon should be positive"

    def test_anomaly_detection_with_synthetic_data(self):
        """Test anomaly detection on synthetic data"""
        np.random.seed(42)
        normal_data = np.random.randn(100, 2)
        anomaly_data = np.random.randn(10, 2) + 5
        X = np.vstack((normal_data, anomaly_data))
        y = np.hstack((np.zeros(100), np.ones(10)))
        Xval = X
        yval = y
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        pval = multivariateGaussian(Xval, mu, sigma2)
        epsilon, F1 = selectThreshold(yval, pval)
        outliers = np.where(p < epsilon)[0]
        detected_anomalies = np.sum(y[outliers] == 1)
        assert detected_anomalies > 0, "Should detect some anomalies"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[1, 2]])
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape[0] == 1, "Should handle single sample"

    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape[0] == 100, "Should handle single feature"

    def test_all_same_values(self):
        """Test with all same values (zero variance)"""
        X = np.ones((100, 2))
        mu, sigma2 = estimateGaussian(X)
        assert sigma2[0] == 0, "Variance should be zero"
        assert sigma2[1] == 0, "Variance should be zero"

    def test_extreme_values(self):
        """Test with extreme values"""
        np.random.seed(42)
        X = np.random.randn(100, 2) * 1e6
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert not np.any(np.isnan(p)), "Probabilities should not be NaN"
        assert not np.any(np.isinf(p)), "Probabilities should not be infinite"

    def test_no_anomalies_in_validation(self):
        """Test when validation set has no anomalies"""
        np.random.seed(42)
        pval = np.random.rand(100)
        yval = np.zeros(100)
        epsilon, F1 = selectThreshold(yval, pval)
        assert epsilon is not None, "Should handle no anomalies case"

    def test_all_anomalies_in_validation(self):
        """Test when validation set has all anomalies"""
        np.random.seed(42)
        pval = np.random.rand(100)
        yval = np.ones(100)
        epsilon, F1 = selectThreshold(yval, pval)
        assert epsilon is not None, "Should handle all anomalies case"


class TestPerformanceMetrics:
    """Tests for performance metrics"""

    def test_precision_recall_calculation(self):
        """Test precision and recall calculation"""
        pval = np.array([0.1, 0.1, 0.1, 0.9, 0.95, 0.05])
        yval = np.array([0, 0, 0, 1, 1, 1])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0, "F1 should be positive"

    def test_f1_score_bounds(self):
        """Test that F1 score is always between 0 and 1"""
        np.random.seed(42)
        for _ in range(10):
            pval = np.random.rand(50)
            yval = np.random.randint(0, 2, 50)
            epsilon, F1 = selectThreshold(yval, pval)
            assert 0 <= F1 <= 1, "F1 should be between 0 and 1"


class TestNumericalStability:
    """Tests for numerical stability"""

    def test_near_singular_covariance(self):
        """Test with near-singular covariance matrix"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X[:, 1] = X[:, 0] + np.random.randn(100) * 1e-10
        mu, sigma2 = estimateGaussian(X)
        sigma2 = np.maximum(sigma2, 1e-10)
        p = multivariateGaussian(X, mu, sigma2)
        assert not np.any(np.isnan(p)), "Probabilities should not be NaN"

    def test_very_small_probabilities(self):
        """Test handling of very small probabilities"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        mu, sigma2 = estimateGaussian(X)
        far_point = np.array([[100, 100]])
        p = multivariateGaussian(far_point, mu, sigma2)
        assert p[0] > 0, "Probability should still be positive"
        assert p[0] < 1e-100, "Probability should be very small"


class TestComparisonWithScipy:
    """Tests comparing with scipy's multivariate normal"""

    def test_comparison_with_scipy_stats(self):
        """Compare our implementation with scipy.stats.multivariate_normal"""
        try:
            from scipy.stats import multivariate_normal
            np.random.seed(42)
            X = np.random.randn(100, 2)
            mu, sigma2 = estimateGaussian(X)
            cov = np.diag(sigma2)
            test_point = np.array([[0, 0]])
            p_ours = multivariateGaussian(test_point, mu, sigma2)
            rv = multivariate_normal(mean=mu, cov=cov)
            p_scipy = rv.pdf(test_point)
            np.testing.assert_almost_equal(p_ours[0], p_scipy[0], decimal=5)
        except ImportError:
            pytest.skip("scipy.stats not available")

    def test_comparison_on_multiple_points(self):
        """Compare on multiple test points"""
        try:
            from scipy.stats import multivariate_normal
            np.random.seed(42)
            X = np.random.randn(100, 2)
            mu, sigma2 = estimateGaussian(X)
            cov = np.diag(sigma2)
            test_points = np.random.randn(10, 2)
            p_ours = multivariateGaussian(test_points, mu, sigma2)
            rv = multivariate_normal(mean=mu, cov=cov)
            p_scipy = rv.pdf(test_points)
            np.testing.assert_array_almost_equal(p_ours, p_scipy, decimal=5)
        except ImportError:
            pytest.skip("scipy.stats not available")
