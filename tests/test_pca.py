# -*- coding: utf-8 -*-
"""
Tests for PCA (Principal Component Analysis) algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "PCA"))

import importlib.util
spec = importlib.util.spec_from_file_location("PCA_module", os.path.join(PROJECT_ROOT, "PCA", "PCA.py"))
pca_module = importlib.util.module_from_spec(spec)

import sklearn.decomposition
original_pca = getattr(sklearn.decomposition, 'pca', None)
if not hasattr(sklearn.decomposition, 'pca'):
    sklearn.decomposition.pca = sklearn.decomposition.PCA

spec.loader.exec_module(pca_module)

featureNormalize = pca_module.featureNormalize
projectData = pca_module.projectData
recoverData = pca_module.recoverData
plot_data_2d = pca_module.plot_data_2d


class TestLoadData:
    def test_load_mat_data_success(self, pca_data_path, load_mat_data):
        data = load_mat_data(pca_data_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data

    def test_load_mat_data_shape(self, pca_data_path, load_mat_data):
        data = load_mat_data(pca_data_path)
        X = data["X"]
        assert X.shape[0] > 0
        assert X.shape[1] == 2

    def test_load_faces_data_success(self, pca_faces_data_path, load_mat_data):
        data = load_mat_data(pca_faces_data_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data


class TestFeatureNormalize:
    def test_feature_normalize_output_shape(self, simple_pca_data):
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        assert X_norm.shape == X.shape
        assert mu.shape[0] == X.shape[1]
        assert sigma.shape[0] == X.shape[1]

    def test_feature_normalize_mean_zero(self, simple_pca_data):
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        mean_after = np.mean(X_norm, axis=0)
        assert_array_almost_equal(mean_after, np.zeros(X.shape[1]), decimal=10)

    def test_feature_normalize_std_one(self, simple_pca_data):
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        std_after = np.std(X_norm, axis=0)
        assert_array_almost_equal(std_after, np.ones(X.shape[1]), decimal=10)

    def test_feature_normalize_mu_correct(self, simple_pca_data):
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        expected_mu = np.mean(X, axis=0)
        assert_array_almost_equal(mu, expected_mu, decimal=10)

    def test_feature_normalize_sigma_correct(self, simple_pca_data):
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        expected_sigma = np.std(X, axis=0)
        assert_array_almost_equal(sigma, expected_sigma, decimal=10)

    def test_feature_normalize_single_feature(self):
        X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        X_norm, mu, sigma = featureNormalize(X.copy())
        assert X_norm.shape == X.shape
        assert_almost_equal(mu[0], 3.0)
        assert sigma[0] > 0


class TestProjectData:
    def test_project_data_output_shape(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z = projectData(X_norm, U, K)
        assert Z.shape == (X.shape[0], K)

    def test_project_data_k1(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        assert Z.shape == (X.shape[0], 1)

    def test_project_data_variance_preserved(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = X.shape[1]
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert_array_almost_equal(X_norm, X_rec, decimal=10)


class TestRecoverData:
    def test_recover_data_output_shape(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert X_rec.shape == X_norm.shape

    def test_recover_data_approximation(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = X.shape[1]
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        reconstruction_error = np.mean((X_norm - X_rec) ** 2)
        assert reconstruction_error < 1e-10

    def test_recover_data_k1(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert X_rec.shape == X_norm.shape


class TestPCACovarianceMatrix:
    def test_covariance_matrix_shape(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        assert Sigma.shape == (X.shape[1], X.shape[1])

    def test_covariance_matrix_symmetric(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        assert_array_almost_equal(Sigma, Sigma.T, decimal=10)

    def test_covariance_matrix_positive_semidefinite(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        eigenvalues = np.linalg.eigvalsh(Sigma)
        assert np.all(eigenvalues >= -1e-10)


class TestPCASVD:
    def test_svd_output_shapes(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        assert U.shape == (n, n)
        assert S.shape == (n,)
        assert V.shape == (n, n)

    def test_svd_singular_values_decreasing(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        assert np.all(np.diff(S) <= 0)

    def test_svd_orthogonal_u(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        n = X.shape[1]
        identity = np.dot(U.T, U)
        assert_array_almost_equal(identity, np.eye(n), decimal=10)


class TestPCAIntegration:
    def test_full_pipeline_2d(self, pca_data_path, load_mat_data):
        data = load_mat_data(pca_data_path)
        X = data["X"]
        m = X.shape[0]
        X_norm, mu, sigma = featureNormalize(X.copy())
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert Z.shape == (m, K)
        assert X_rec.shape == X.shape

    def test_variance_retained(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        total_variance = np.sum(S)
        K = min(X.shape[1], 3)
        retained_variance = np.sum(S[:K])
        ratio = retained_variance / total_variance
        assert ratio > 0.4

    def test_reconstruction_error_increases_with_lower_k(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        errors = []
        for K in range(1, X.shape[1] + 1):
            Z = projectData(X_norm, U, K)
            X_rec = recoverData(Z, U, K)
            error = np.mean((X_norm - X_rec) ** 2)
            errors.append(error)
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1]


class TestPCASklearnComparison:
    def test_sklearn_comparison(self, simple_pca_data):
        from sklearn.decomposition import PCA as SklearnPCA
        X = simple_pca_data
        X_norm, mu, sigma = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 2
        Z_custom = projectData(X_norm, U, K)
        sklearn_pca = SklearnPCA(n_components=K)
        Z_sklearn = sklearn_pca.fit_transform(X_norm)
        assert Z_custom.shape == Z_sklearn.shape

    def test_sklearn_variance_ratio(self, simple_pca_data):
        from sklearn.decomposition import PCA as SklearnPCA
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        total_variance = np.sum(S)
        K = 2
        retained_variance = np.sum(S[:K])
        custom_ratio = retained_variance / total_variance
        sklearn_pca = SklearnPCA(n_components=K)
        sklearn_pca.fit(X_norm)
        sklearn_ratio = np.sum(sklearn_pca.explained_variance_ratio_)
        assert_almost_equal(custom_ratio, sklearn_ratio, decimal=2)


class TestEdgeCases:
    def test_single_sample(self):
        X = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])
        X_norm, mu, sigma = featureNormalize(X.copy())
        assert X_norm.shape == X.shape

    def test_single_feature(self):
        X = np.random.randn(100, 1)
        X_norm, mu, sigma = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 1
        Z = projectData(X_norm, U, K)
        assert Z.shape == (100, 1)

    def test_k_equals_n(self, simple_pca_data):
        X = simple_pca_data
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = X.shape[1]
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert_array_almost_equal(X_norm, X_rec, decimal=10)

    def test_high_dimensional_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 50)
        X_norm, _, _ = featureNormalize(X.copy())
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 10
        Z = projectData(X_norm, U, K)
        assert Z.shape == (100, K)


class TestPCAFaceImages:
    def test_face_data_shape(self, pca_faces_data_path, load_mat_data):
        data = load_mat_data(pca_faces_data_path)
        X = data["X"]
        assert X.shape[1] == 1024

    def test_face_data_pca(self, pca_faces_data_path, load_mat_data):
        data = load_mat_data(pca_faces_data_path)
        X = data["X"][:100, :]
        m = X.shape[0]
        X_norm, mu, sigma = featureNormalize(X.copy())
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 100
        Z = projectData(X_norm, U, K)
        assert Z.shape == (100, K)
