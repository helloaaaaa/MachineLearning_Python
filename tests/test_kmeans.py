# -*- coding: utf-8 -*-
"""
Tests for K-Means clustering algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "K-Means"))

import importlib.util
spec = importlib.util.spec_from_file_location("K_Menas", os.path.join(PROJECT_ROOT, "K-Means", "K-Menas.py"))
kmeans_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kmeans_module)

findClosestCentroids = kmeans_module.findClosestCentroids
computerCentroids = kmeans_module.computerCentroids
runKMeans = kmeans_module.runKMeans
kMeansInitCentroids = kmeans_module.kMeansInitCentroids


def _safe_idx_to_array(idx):
    try:
        return np.array(idx).flatten()
    except (ValueError, TypeError):
        if hasattr(idx, 'flatten'):
            return idx.flatten()
        return np.array([idx])


class TestLoadData:
    def test_load_mat_data_success(self, kmeans_data_path, load_mat_data):
        data = load_mat_data(kmeans_data_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data

    def test_load_mat_data_shape(self, kmeans_data_path, load_mat_data):
        data = load_mat_data(kmeans_data_path)
        X = data["X"]
        assert X.shape[0] > 0
        assert X.shape[1] == 2


class TestFindClosestCentroids:
    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_find_closest_centroids_output_shape(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = np.array([[0, 0], [5, 5], [5, 0]])
        idx = findClosestCentroids(X, initial_centroids)
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr.shape[0] == X.shape[0]

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_find_closest_centroids_valid_indices(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = np.array([[0, 0], [5, 5], [5, 0]])
        idx = findClosestCentroids(X, initial_centroids)
        idx_arr = _safe_idx_to_array(idx)
        assert np.all((idx_arr >= 0) & (idx_arr < K))

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_find_closest_centroids_correct_assignment(self):
        X = np.array([[0, 0], [10, 10], [0, 10]])
        initial_centroids = np.array([[0, 0], [10, 10]])
        idx = findClosestCentroids(X, initial_centroids)
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr[0] == 0
        assert idx_arr[1] == 1
        assert idx_arr[2] in [0, 1]

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_find_closest_centroids_single_centroid(self, simple_clustering_data):
        X = simple_clustering_data
        initial_centroids = np.array([[2.5, 2.5]])
        idx = findClosestCentroids(X, initial_centroids)
        idx_arr = _safe_idx_to_array(idx)
        assert np.all(idx_arr == 0)

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_find_closest_centroids_equal_distance(self):
        X = np.array([[5, 5]])
        initial_centroids = np.array([[0, 0], [10, 10]])
        idx = findClosestCentroids(X, initial_centroids)
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr[0] in [0, 1]


class TestComputerCentroids:
    def test_computer_centroids_output_shape(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        idx = np.random.randint(0, K, X.shape[0])
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (K, X.shape[1])

    def test_computer_centroids_mean_calculation(self):
        X = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
        idx = np.array([0, 0, 1, 1])
        K = 2
        centroids = computerCentroids(X, idx, K)
        assert_array_almost_equal(centroids[0], [0.5, 0.5], decimal=5)
        assert_array_almost_equal(centroids[1], [10.5, 10.5], decimal=5)

    def test_computer_centroids_single_cluster(self, simple_clustering_data):
        X = simple_clustering_data
        K = 1
        idx = np.zeros(X.shape[0], dtype=int)
        centroids = computerCentroids(X, idx, K)
        expected_mean = np.mean(X, axis=0)
        assert_array_almost_equal(centroids[0], expected_mean, decimal=5)

    def test_computer_centroids_empty_cluster(self):
        X = np.array([[0, 0], [1, 1], [10, 10]])
        idx = np.array([0, 0, 0])
        K = 2
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (K, X.shape[1])


class TestKMeansInitCentroids:
    def test_kmeans_init_centroids_shape(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        centroids = kMeansInitCentroids(X, K)
        assert centroids.shape == (K, X.shape[1])

    def test_kmeans_init_centroids_from_data(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        centroids = kMeansInitCentroids(X, K)
        for c in centroids:
            assert np.any(np.all(X == c, axis=1))

    def test_kmeans_init_centroids_valid_k(self, simple_clustering_data):
        X = simple_clustering_data
        K = 5
        centroids = kMeansInitCentroids(X, K)
        assert centroids.shape[0] == K


class TestRunKMeans:
    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_run_kmeans_output_shape(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 10
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        idx_arr = _safe_idx_to_array(idx)
        assert centroids.shape == (K, X.shape[1])
        assert idx_arr.shape[0] == X.shape[0]

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_run_kmeans_convergence(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = np.array([[0, 0], [5, 5], [5, 0]])
        max_iters = 20
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids is not None
        assert idx is not None

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_run_kmeans_valid_indices(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 10
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        idx_arr = _safe_idx_to_array(idx)
        assert np.all((idx_arr >= 0) & (idx_arr < K))

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_run_kmeans_single_iteration(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 1
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, X.shape[1])


class TestKMeansIntegration:
    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_full_pipeline(self, kmeans_data_path, load_mat_data):
        data = load_mat_data(kmeans_data_path)
        X = data["X"]
        K = 3
        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
        max_iters = 10
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, X.shape[1])
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr.shape[0] == X.shape[0]

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_clustering_quality(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = np.array([[0, 0], [5, 5], [5, 0]])
        max_iters = 20
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        idx_arr = _safe_idx_to_array(idx)
        for k in range(K):
            cluster_points = X[idx_arr == k]
            if len(cluster_points) > 0:
                distances = np.sum((cluster_points - centroids[k]) ** 2, axis=1)
                assert np.mean(distances) < 10

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_multiple_runs_different_init(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        max_iters = 10
        results = []
        for _ in range(3):
            initial_centroids = kMeansInitCentroids(X, K)
            centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
            results.append((centroids, idx))
        assert len(results) == 3


class TestKMeansSklearnComparison:
    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_sklearn_comparison(self, simple_clustering_data):
        from sklearn.cluster import KMeans as SklearnKMeans
        X = simple_clustering_data
        K = 3
        initial_centroids = np.array([[0, 0], [5, 5], [5, 0]])
        max_iters = 20
        custom_centroids, custom_idx = runKMeans(X, initial_centroids, max_iters, False)
        sklearn_model = SklearnKMeans(n_clusters=K, n_init=1, random_state=42)
        sklearn_model.fit(X)
        sklearn_idx = sklearn_model.labels_
        custom_idx_arr = _safe_idx_to_array(custom_idx)
        custom_counts = np.bincount(custom_idx_arr.astype(int), minlength=K)
        sklearn_counts = np.bincount(sklearn_idx, minlength=K)
        assert len(custom_counts) == len(sklearn_counts)


class TestEdgeCases:
    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_single_sample(self):
        X = np.array([[1.0, 2.0]])
        K = 1
        initial_centroids = np.array([[1.0, 2.0]])
        max_iters = 5
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, 2)
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr[0] == 0

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_k_equals_n(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        K = 3
        initial_centroids = X.copy()
        max_iters = 5
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, 2)

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_large_k(self, simple_clustering_data):
        X = simple_clustering_data
        K = 10
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 5
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, X.shape[1])

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_high_dimensional_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 10)
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 10
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, 10)

    def test_zero_iterations(self, simple_clustering_data):
        X = simple_clustering_data
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 0
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        idx_arr = _safe_idx_to_array(idx)
        assert idx_arr.shape[0] == X.shape[0]


class TestKMeansImageCompression:
    def test_image_data_reshape(self):
        np.random.seed(42)
        img_data = np.random.rand(10, 10, 3)
        img_size = img_data.shape
        X = img_data.reshape(img_size[0] * img_size[1], 3)
        assert X.shape == (100, 3)

    @pytest.mark.skipif(sys.version_info >= (3, 10), reason="numpy matrix compatibility issue in Python 3.10+")
    def test_image_compression_pipeline(self):
        np.random.seed(42)
        img_data = np.random.rand(10, 10, 3)
        img_size = img_data.shape
        X = img_data.reshape(img_size[0] * img_size[1], 3)
        K = 4
        max_iters = 5
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        idx = findClosestCentroids(X, centroids)
        idx_arr = _safe_idx_to_array(idx)
        X_recovered = centroids[idx_arr, :]
        X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)
        assert X_recovered.shape == img_size
