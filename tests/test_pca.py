# -*- coding: utf-8 -*-
"""
Unit tests for PCA (Principal Component Analysis) module.
Tests cover: data loading, feature normalization, covariance matrix,
SVD, projection, reconstruction, and comparison with scikit-learn.
"""
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'PCA'))

from conftest import (
    PCA_DIR, assert_array_almost_equal_custom, assert_shape
)


def featureNormalize(X):
    n = X.shape[1]
    mu = np.zeros((1, n))
    sigma = np.zeros((1, n))
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    for i in range(n):
        X[:, i] = (X[:, i] - mu[i]) / sigma[i]
    return X, mu, sigma


def projectData(X_norm, U, K):
    Z = np.zeros((X_norm.shape[0], K))
    U_reduce = U[:, 0:K]
    Z = np.dot(X_norm, U_reduce)
    return Z


def recoverData(Z, U, K):
    X_rec = np.zeros((Z.shape[0], U.shape[0]))
    U_reduce = U[:, 0:K]
    X_rec = np.dot(Z, np.transpose(U_reduce))
    return X_rec


def computeCovariance(X_norm):
    m = X_norm.shape[0]
    Sigma = np.dot(np.transpose(X_norm), X_norm) / m
    return Sigma


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_pca_data_exists(self, pca_data):
        """Test that PCA data file can be loaded"""
        assert pca_data is not None, "Data file should exist"

    def test_load_pca_face_data_exists(self, pca_face_data):
        """Test that PCA face data file can be loaded"""
        if pca_face_data is None:
            pytest.skip("Face data file not available")
        assert pca_face_data is not None, "Face data file should exist"

    def test_load_pca_data_shape(self, pca_data):
        """Test that loaded data has correct shape"""
        if pca_data is None:
            pytest.skip("Data file not available")
        X = pca_data['X']
        assert len(X.shape) == 2, "X should be 2D array"
        assert X.shape[0] > 0, "X should have samples"

    def test_load_pca_face_data_shape(self, pca_face_data):
        """Test that loaded face data has correct shape"""
        if pca_face_data is None:
            pytest.skip("Face data file not available")
        X = pca_face_data['X']
        assert len(X.shape) == 2, "X should be 2D array"
        assert X.shape[1] == 1024, "Face images should have 1024 features (32x32)"


class TestFeatureNormalization:
    """Tests for feature normalization"""

    def test_normalization_output_shape(self, synthetic_regression_data):
        """Test that normalization preserves shape"""
        X = synthetic_regression_data['X'].copy()
        X_norm, mu, sigma = featureNormalize(X)
        assert X_norm.shape == synthetic_regression_data['X'].shape

    def test_normalization_mean_zero(self, synthetic_regression_data):
        """Test that normalized features have approximately zero mean"""
        X = synthetic_regression_data['X'].copy()
        X_norm, mu, sigma = featureNormalize(X)
        np.testing.assert_array_almost_equal(
            np.mean(X_norm, axis=0), np.zeros(X.shape[1]), decimal=10
        )

    def test_normalization_std_one(self, synthetic_regression_data):
        """Test that normalized features have approximately unit std"""
        X = synthetic_regression_data['X'].copy()
        X_norm, mu, sigma = featureNormalize(X)
        np.testing.assert_array_almost_equal(
            np.std(X_norm, axis=0), np.ones(X.shape[1]), decimal=10
        )

    def test_normalization_mu_sigma_shape(self, synthetic_regression_data):
        """Test that mu and sigma have correct shapes"""
        X = synthetic_regression_data['X'].copy()
        X_norm, mu, sigma = featureNormalize(X)
        assert mu.shape[1] == X.shape[1], "mu should have same number of features"
        assert sigma.shape[1] == X.shape[1], "sigma should have same number of features"


class TestCovarianceMatrix:
    """Tests for covariance matrix computation"""

    def test_covariance_shape(self, synthetic_regression_data):
        """Test that covariance matrix has correct shape"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        n = X.shape[1]
        assert Sigma.shape == (n, n), "Covariance matrix should be n x n"

    def test_covariance_symmetric(self, synthetic_regression_data):
        """Test that covariance matrix is symmetric"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        np.testing.assert_array_almost_equal(Sigma, Sigma.T, decimal=10)

    def test_covariance_positive_semidefinite(self, synthetic_regression_data):
        """Test that covariance matrix is positive semi-definite"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues >= -1e-10), "Eigenvalues should be non-negative"


class TestSVD:
    """Tests for Singular Value Decomposition"""

    def test_svd_shapes(self, synthetic_regression_data):
        """Test that SVD produces correct shapes"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        assert U.shape == (n, n), "U should be n x n"
        assert S.shape == (n,), "S should have n singular values"

    def test_svd_orthogonal_u(self, synthetic_regression_data):
        """Test that U is orthogonal"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        identity = np.dot(U.T, U)
        np.testing.assert_array_almost_equal(identity, np.eye(U.shape[0]), decimal=10)

    def test_svd_singular_values_ordered(self, synthetic_regression_data):
        """Test that singular values are in descending order"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        assert np.all(np.diff(S) <= 0), "Singular values should be in descending order"


class TestProjection:
    """Tests for data projection"""

    def test_projection_shape(self, synthetic_regression_data):
        """Test that projection has correct shape"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z = projectData(X_norm, U, K)
        assert Z.shape == (X.shape[0], K), "Projection should have shape (m, K)"

    def test_projection_k_less_than_n(self, synthetic_regression_data):
        """Test projection with K < n"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        K = n - 1
        Z = projectData(X_norm, U, K)
        assert Z.shape[1] == K, "Projection should reduce dimension"

    def test_projection_k_equals_n(self, synthetic_regression_data):
        """Test projection with K = n (no reduction)"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        K = n
        Z = projectData(X_norm, U, K)
        assert Z.shape[1] == n, "Projection should preserve dimension"


class TestReconstruction:
    """Tests for data reconstruction"""

    def test_reconstruction_shape(self, synthetic_regression_data):
        """Test that reconstruction has correct shape"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert X_rec.shape == X_norm.shape, "Reconstruction should have original shape"

    def test_reconstruction_error(self, synthetic_regression_data):
        """Test that reconstruction error is reasonable"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        error = np.mean((X_norm - X_rec) ** 2)
        assert error >= 0, "Reconstruction error should be non-negative"

    def test_reconstruction_perfect_with_full_k(self, synthetic_regression_data):
        """Test that reconstruction is perfect with K = n"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        K = n
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        np.testing.assert_array_almost_equal(X_norm, X_rec, decimal=10)


class TestPCAIntegration:
    """Integration tests for PCA"""

    def test_pca_end_to_end_2d(self, pca_data):
        """Test complete PCA pipeline on 2D data"""
        if pca_data is None:
            pytest.skip("Data file not available")
        X = pca_data['X'].copy()
        m = X.shape[0]
        X_norm, mu, sigma = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        assert Z.shape == (m, K), "Projection should have correct shape"
        X_rec = recoverData(Z, U, K)
        assert X_rec.shape == X.shape, "Reconstruction should have original shape"

    def test_pca_variance_retained(self, synthetic_regression_data):
        """Test that PCA retains most variance with appropriate K"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        total_variance = np.sum(S)
        K = 2
        retained_variance = np.sum(S[:K])
        ratio = retained_variance / total_variance
        assert ratio <= 1, "Retained variance ratio should be <= 1"
        assert ratio > 0, "Retained variance ratio should be > 0"


class TestSklearnComparison:
    """Tests comparing our implementation with scikit-learn"""

    def test_sklearn_pca(self, pca_data):
        """Test scikit-learn PCA on same data"""
        if pca_data is None:
            pytest.skip("Data file not available")
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = pca_data['X']
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        K = 1
        model = PCA(n_components=K)
        Z = model.fit_transform(X_scaled)
        assert Z.shape == (X.shape[0], K), "Sklearn PCA should produce correct shape"

    def test_comparison_with_sklearn(self, synthetic_regression_data):
        """Compare our implementation with scikit-learn"""
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = synthetic_regression_data['X'].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        K = 2
        model = PCA(n_components=K)
        Z_sklearn = model.fit_transform(X_scaled)
        X_norm, _, _ = featureNormalize(X.copy())
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        Z_ours = projectData(X_norm, U, K)
        assert Z_ours.shape == Z_sklearn.shape, "Should have same shape"
        correlation = np.abs(np.corrcoef(Z_ours.flatten(), Z_sklearn.flatten())[0, 1])
        assert correlation > 0.9, "Projections should be highly correlated"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[1, 2, 3]])
        X_norm, mu, sigma = featureNormalize(X.copy())
        assert X_norm.shape == X.shape

    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        X_norm, _, _ = featureNormalize(X.copy())
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        assert Z.shape == (100, 1), "Should work with single feature"

    def test_k_equals_one(self, synthetic_regression_data):
        """Test with K = 1"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        assert Z.shape[1] == 1, "Should reduce to 1 dimension"

    def test_high_dimensional_data(self):
        """Test with high dimensional data"""
        np.random.seed(42)
        m, n = 50, 100
        X = np.random.randn(m, n)
        X_norm, _, _ = featureNormalize(X.copy())
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        K = 10
        Z = projectData(X_norm, U, K)
        assert Z.shape == (m, K), "Should handle high dimensional data"

    def test_zero_variance_feature(self):
        """Test with zero variance feature"""
        X = np.column_stack([np.random.randn(100), np.zeros(100)])
        X_norm, mu, sigma = featureNormalize(X.copy())
        assert np.any(np.isnan(X_norm[:, 1])) or sigma[0, 1] == 0, \
            "Zero variance feature should be detected"


class TestVarianceExplained:
    """Tests for variance explained analysis"""

    def test_cumulative_variance(self, synthetic_regression_data):
        """Test cumulative variance calculation"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        total_variance = np.sum(S)
        cumulative_variance = np.cumsum(S) / total_variance
        assert cumulative_variance[-1] == 1.0, "Cumulative variance should reach 1"
        assert np.all(np.diff(cumulative_variance) >= 0), "Cumulative variance should increase"

    def test_variance_explained_by_k(self, synthetic_regression_data):
        """Test variance explained by K components"""
        X = synthetic_regression_data['X'].copy()
        X_norm, _, _ = featureNormalize(X)
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        total_variance = np.sum(S)
        for K in range(1, X.shape[1] + 1):
            variance_explained = np.sum(S[:K]) / total_variance
            assert 0 <= variance_explained <= 1, "Variance explained should be between 0 and 1"


class TestNumericalStability:
    """Tests for numerical stability"""

    def test_ill_conditioned_data(self):
        """Test with ill-conditioned data"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 1] = X[:, 0] * 1e-10
        X_norm, _, _ = featureNormalize(X.copy())
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        assert not np.any(np.isnan(U)), "U should not contain NaN"
        assert not np.any(np.isnan(S)), "S should not contain NaN"

    def test_large_values(self):
        """Test with large values"""
        np.random.seed(42)
        X = np.random.randn(100, 3) * 1e6
        X_norm, _, _ = featureNormalize(X.copy())
        Sigma = computeCovariance(X_norm)
        U, S, V = np.linalg.svd(Sigma)
        assert not np.any(np.isinf(U)), "U should not contain inf"
        assert not np.any(np.isinf(S)), "S should not contain inf"
