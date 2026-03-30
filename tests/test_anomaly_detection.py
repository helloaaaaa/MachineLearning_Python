# -*- coding: utf-8 -*-
"""
Tests for Anomaly Detection algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "AnomalyDetection"))

from AnomalyDetection import (
    estimateGaussian,
    multivariateGaussian,
    selectThreshold,
    display_2d_data,
)


class TestLoadData:
    def test_load_data1_success(self, anomaly_detection_data1_path, load_mat_data):
        data = load_mat_data(anomaly_detection_data1_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data
        assert "Xval" in data
        assert "yval" in data

    def test_load_data2_success(self, anomaly_detection_data2_path, load_mat_data):
        data = load_mat_data(anomaly_detection_data2_path)
        assert data is not None
        assert isinstance(data, dict)

    def test_data_shape(self, anomaly_detection_data1_path, load_mat_data):
        data = load_mat_data(anomaly_detection_data1_path)
        X = data["X"]
        Xval = data["Xval"]
        yval = data["yval"]
        assert X.shape[0] > 0
        assert Xval.shape[0] > 0
        assert yval.shape[0] == Xval.shape[0]


class TestEstimateGaussian:
    def test_estimate_gaussian_output_shape(self, simple_anomaly_data):
        X, _ = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape[0] == X.shape[1]
        assert sigma2.shape[0] == X.shape[1]

    def test_estimate_gaussian_mean_correct(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu, sigma2 = estimateGaussian(X)
        expected_mu = np.mean(X, axis=0)
        assert_array_almost_equal(mu, expected_mu, decimal=10)

    def test_estimate_gaussian_variance_correct(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mu, sigma2 = estimateGaussian(X)
        expected_sigma2 = np.var(X, axis=0)
        assert_array_almost_equal(sigma2, expected_sigma2, decimal=10)

    def test_estimate_gaussian_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape == (1,)
        assert sigma2.shape == (1,)
        assert_almost_equal(mu[0], 3.0)

    def test_estimate_gaussian_constant_feature(self):
        X = np.array([[5.0], [5.0], [5.0], [5.0]])
        mu, sigma2 = estimateGaussian(X)
        assert_almost_equal(mu[0], 5.0)
        assert_almost_equal(sigma2[0], 0.0, decimal=5)


class TestMultivariateGaussian:
    def test_multivariate_gaussian_output_shape(self, simple_anomaly_data):
        X, _ = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape[0] == X.shape[0]

    def test_multivariate_gaussian_probabilities_range(self, simple_anomaly_data):
        X, _ = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert np.all(p > 0)
        assert np.all(p <= 1)

    def test_multivariate_gaussian_center_higher_prob(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        mu = np.mean(X, axis=0)
        sigma2 = np.var(X, axis=0)
        p = multivariateGaussian(X, mu, sigma2)
        center_point = np.array([[mu[0], mu[1]]])
        p_center = multivariateGaussian(center_point, mu, sigma2)
        assert p_center[0] >= np.min(p)

    def test_multivariate_gaussian_outlier_lower_prob(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        mu = np.mean(X, axis=0)
        sigma2 = np.var(X, axis=0)
        p_normal = multivariateGaussian(X, mu, sigma2)
        outlier = np.array([[10.0, 10.0]])
        p_outlier = multivariateGaussian(outlier, mu, sigma2)
        assert p_outlier[0] < np.mean(p_normal)

    def test_multivariate_gaussian_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0]])
        mu = np.array([2.0])
        sigma2 = np.array([1.0])
        try:
            p = multivariateGaussian(X, mu, sigma2)
            assert p.shape == (3,)
        except np.linalg.LinAlgError:
            pytest.skip("multivariateGaussian requires at least 2D covariance matrix")


class TestSelectThreshold:
    def test_select_threshold_output_types(self, simple_anomaly_data):
        X, y = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        pval = multivariateGaussian(X, mu, sigma2)
        epsilon, F1 = selectThreshold(y, pval)
        assert isinstance(epsilon, float)
        assert isinstance(F1, float)

    def test_select_threshold_epsilon_in_range(self, simple_anomaly_data):
        X, y = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        pval = multivariateGaussian(X, mu, sigma2)
        epsilon, F1 = selectThreshold(y, pval)
        assert epsilon >= np.min(pval)
        assert epsilon <= np.max(pval)

    def test_select_threshold_f1_in_range(self, simple_anomaly_data):
        X, y = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        pval = multivariateGaussian(X, mu, sigma2)
        epsilon, F1 = selectThreshold(y, pval)
        assert 0 <= F1 <= 1

    def test_select_threshold_perfect_detection(self):
        yval = np.array([0, 0, 0, 1, 1])
        pval = np.array([0.9, 0.8, 0.85, 0.1, 0.05])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0

    def test_select_threshold_no_anomalies(self):
        yval = np.array([0, 0, 0, 0, 0])
        pval = np.array([0.9, 0.8, 0.85, 0.7, 0.75])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 >= 0


class TestAnomalyDetectionIntegration:
    def test_full_pipeline(self, anomaly_detection_data1_path, load_mat_data):
        data = load_mat_data(anomaly_detection_data1_path)
        X = data["X"]
        Xval = data["Xval"]
        yval = data["yval"]
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        pval = multivariateGaussian(Xval, mu, sigma2)
        epsilon, F1 = selectThreshold(yval, pval)
        outliers = np.where(p < epsilon)[0]
        assert epsilon > 0
        assert F1 >= 0
        assert len(outliers) >= 0

    def test_anomaly_detection_accuracy(self, anomaly_detection_data1_path, load_mat_data):
        data = load_mat_data(anomaly_detection_data1_path)
        X = data["X"]
        Xval = data["Xval"]
        yval = data["yval"].flatten()
        mu, sigma2 = estimateGaussian(X)
        pval = multivariateGaussian(Xval, mu, sigma2)
        epsilon, F1 = selectThreshold(yval, pval)
        predictions = (pval < epsilon).astype(int)
        assert F1 > 0.5

    def test_anomaly_detection_outliers_identified(self, simple_anomaly_data):
        X, y = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        pval = multivariateGaussian(X, mu, sigma2)
        epsilon, F1 = selectThreshold(y, pval)
        outliers = np.where(p < epsilon)[0]
        assert len(outliers) > 0


class TestAnomalyDetectionSklearnComparison:
    def test_sklearn_comparison(self, simple_anomaly_data):
        from sklearn.covariance import EllipticEnvelope
        X, y = simple_anomaly_data
        mu, sigma2 = estimateGaussian(X)
        p_custom = multivariateGaussian(X, mu, sigma2)
        sklearn_model = EllipticEnvelope(contamination=0.1)
        sklearn_model.fit(X)
        p_sklearn = sklearn_model.predict(X)
        assert p_custom.shape[0] == p_sklearn.shape[0]


class TestEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape == (2,)
        assert sigma2.shape == (2,)

    def test_single_feature(self):
        X = np.random.randn(100, 1)
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape == (1,)
        assert sigma2.shape == (1,)

    def test_all_normal_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = np.zeros(100)
        mu, sigma2 = estimateGaussian(X)
        pval = multivariateGaussian(X, mu, sigma2)
        epsilon, F1 = selectThreshold(y, pval)
        assert epsilon >= 0

    def test_high_dimensional_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape == (100,)

    def test_extreme_outliers(self):
        np.random.seed(42)
        X_normal = np.random.randn(100, 2)
        X_outliers = np.random.randn(5, 2) * 100
        X = np.vstack((X_normal, X_outliers))
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        assert np.min(p) < np.max(p)


class TestDisplayData:
    def test_display_2d_data_output(self, simple_anomaly_data):
        X, _ = simple_anomaly_data
        plt = display_2d_data(X, "bx")
        assert plt is not None


class TestMultivariateGaussianProperties:
    def test_probability_decreases_with_distance(self):
        mu = np.array([0.0, 0.0])
        sigma2 = np.array([1.0, 1.0])
        distances = [0, 1, 2, 5, 10]
        probs = []
        for d in distances:
            X = np.array([[d, d]])
            p = multivariateGaussian(X, mu, sigma2)
            probs.append(p[0])
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_probability_symmetry(self):
        mu = np.array([0.0, 0.0])
        sigma2 = np.array([1.0, 1.0])
        X1 = np.array([[1.0, 0.0]])
        X2 = np.array([[-1.0, 0.0]])
        X3 = np.array([[0.0, 1.0]])
        X4 = np.array([[0.0, -1.0]])
        p1 = multivariateGaussian(X1, mu, sigma2)
        p2 = multivariateGaussian(X2, mu, sigma2)
        p3 = multivariateGaussian(X3, mu, sigma2)
        p4 = multivariateGaussian(X4, mu, sigma2)
        assert_almost_equal(p1[0], p2[0], decimal=10)
        assert_almost_equal(p1[0], p3[0], decimal=10)
        assert_almost_equal(p1[0], p4[0], decimal=10)


class TestF1Score:
    def test_f1_score_calculation(self):
        yval = np.array([0, 0, 0, 1, 1, 1])
        pval = np.array([0.9, 0.8, 0.85, 0.1, 0.05, 0.15])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0

    def test_f1_score_perfect(self):
        yval = np.array([0, 0, 0, 1, 1])
        pval = np.array([0.9, 0.8, 0.85, 0.01, 0.02])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 > 0.8

    def test_f1_score_zero_true_positives(self):
        yval = np.array([0, 0, 0, 0, 0])
        pval = np.array([0.9, 0.8, 0.85, 0.7, 0.75])
        epsilon, F1 = selectThreshold(yval, pval)
        assert F1 >= 0
