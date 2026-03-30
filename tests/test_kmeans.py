# -*- coding: utf-8 -*-
"""
Unit tests for K-Means clustering module.
Tests cover: data loading, centroid initialization, closest centroid finding,
centroid computation, clustering algorithm, and comparison with scikit-learn.
"""
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'K-Means'))

from conftest import (
    KMEANS_DIR, assert_array_almost_equal_custom, assert_shape
)


def findClosestCentroids(X, initial_centroids):
    m = X.shape[0]
    K = initial_centroids.shape[0]
    dis = np.zeros((m, K))
    idx = np.zeros((m, 1))
    for i in range(m):
        for j in range(K):
            dis[i, j] = np.dot((X[i, :] - initial_centroids[j, :]).reshape(1, -1), (X[i, :] - initial_centroids[j, :]).reshape(-1, 1))
    dummy, idx = np.where(dis == np.min(dis, axis=1).reshape(-1, 1))
    return idx[0:dis.shape[0]]


def computerCentroids(X, idx, K):
    n = X.shape[1]
    centroids = np.zeros((K, n))
    for i in range(K):
        centroids[i, :] = np.mean(X[np.ravel(idx == i), :], axis=0).reshape(1, -1)
    return centroids


def kMeansInitCentroids(X, K):
    m = X.shape[0]
    m_arr = np.arange(0, m)
    centroids = np.zeros((K, X.shape[1]))
    np.random.shuffle(m_arr)
    rand_indices = m_arr[:K]
    centroids = X[rand_indices, :]
    return centroids


def runKMeans(X, initial_centroids, max_iters, plot_process=False):
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))
    for i in range(max_iters):
        idx = findClosestCentroids(X, centroids)
        previous_centroids = centroids
        centroids = computerCentroids(X, idx, K)
    return centroids, idx


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_kmeans_data_exists(self, kmeans_data):
        """Test that K-Means data file can be loaded"""
        assert kmeans_data is not None, "Data file should exist"

    def test_load_kmeans_data_shape(self, kmeans_data):
        """Test that loaded data has correct shape"""
        if kmeans_data is None:
            pytest.skip("Data file not available")
        X = kmeans_data['X']
        assert len(X.shape) == 2, "X should be 2D array"
        assert X.shape[0] > 0, "X should have samples"
        assert X.shape[1] == 2, "X should have 2 features"

    def test_load_kmeans_data_values(self, kmeans_data):
        """Test that loaded data values are valid"""
        if kmeans_data is None:
            pytest.skip("Data file not available")
        X = kmeans_data['X']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isinf(X)), "X should not contain infinite values"


class TestFindClosestCentroids:
    """Tests for finding closest centroids"""

    def test_find_closest_centroids_shape(self, synthetic_clustering_data):
        """Test that closest centroid indices have correct shape"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = np.array([[0, 0], [5, 5], [0, 5]])
        idx = findClosestCentroids(X, initial_centroids)
        assert idx.shape[0] == X.shape[0], "Should have index for each sample"

    def test_find_closest_centroids_valid_indices(self, synthetic_clustering_data):
        """Test that indices are valid cluster indices"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = np.array([[0, 0], [5, 5], [0, 5]])
        idx = findClosestCentroids(X, initial_centroids)
        assert np.all(idx >= 0), "Indices should be non-negative"
        assert np.all(idx < K), "Indices should be less than K"

    def test_find_closest_centroids_correct_assignment(self):
        """Test that points are assigned to nearest centroid"""
        X = np.array([[0, 0], [10, 10], [0, 10]])
        centroids = np.array([[0, 0], [10, 10], [0, 10]])
        idx = findClosestCentroids(X, centroids)
        expected = np.array([0, 1, 2])
        np.testing.assert_array_equal(idx, expected)

    def test_find_closest_centroids_single_centroid(self):
        """Test with single centroid"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        centroids = np.array([[0, 0]])
        idx = findClosestCentroids(X, centroids)
        assert np.all(idx == 0), "All points should be assigned to single centroid"


class TestComputeCentroids:
    """Tests for computing centroids"""

    def test_compute_centroids_shape(self, synthetic_clustering_data):
        """Test that computed centroids have correct shape"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        idx = np.random.randint(0, K, X.shape[0])
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (K, X.shape[1]), "Centroids should have shape (K, n_features)"

    def test_compute_centroids_mean(self):
        """Test that centroids are mean of assigned points"""
        X = np.array([[0, 0], [2, 2], [10, 10], [12, 12]])
        idx = np.array([0, 0, 1, 1])
        K = 2
        centroids = computerCentroids(X, idx, K)
        expected = np.array([[1, 1], [11, 11]])
        np.testing.assert_array_almost_equal(centroids, expected)

    def test_compute_centroids_empty_cluster(self):
        """Test handling of empty cluster"""
        X = np.array([[0, 0], [2, 2], [10, 10]])
        idx = np.array([0, 0, 1])
        K = 3
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (3, 2), "Should still return K centroids"


class TestKMeansInitCentroids:
    """Tests for K-Means centroid initialization"""

    def test_init_centroids_shape(self, synthetic_clustering_data):
        """Test that initialized centroids have correct shape"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        centroids = kMeansInitCentroids(X, K)
        assert centroids.shape == (K, X.shape[1]), "Centroids should have shape (K, n_features)"

    def test_init_centroids_from_data(self, synthetic_clustering_data):
        """Test that initialized centroids are from the data"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        centroids = kMeansInitCentroids(X, K)
        for centroid in centroids:
            distances = np.sqrt(np.sum((X - centroid) ** 2, axis=1))
            assert np.min(distances) < 1e-10, "Each centroid should be a data point"

    def test_init_centroids_different(self, synthetic_clustering_data):
        """Test that initialized centroids are different (with high probability)"""
        X = synthetic_clustering_data['X']
        K = 3
        centroids = kMeansInitCentroids(X, K)
        assert len(np.unique(centroids, axis=0)) == K, "Centroids should be unique"


class TestRunKMeans:
    """Tests for running K-Means algorithm"""

    def test_run_kmeans_output_shape(self, synthetic_clustering_data):
        """Test that K-Means outputs have correct shapes"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert centroids.shape == (K, X.shape[1]), "Centroids should have shape (K, n_features)"
        assert idx.shape[0] == X.shape[0], "Indices should have same count as samples"

    def test_run_kmeans_convergence(self, synthetic_clustering_data):
        """Test that K-Means converges (centroids stabilize)"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids1, _ = runKMeans(X, initial_centroids, max_iters=10)
        centroids2, _ = runKMeans(X, centroids1, max_iters=10)
        np.testing.assert_array_almost_equal(centroids1, centroids2, decimal=5)

    def test_run_kmeans_cluster_assignment(self, synthetic_clustering_data):
        """Test that K-Means assigns all points to clusters"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert np.all(idx >= 0), "All indices should be non-negative"
        assert np.all(idx < K), "All indices should be less than K"


class TestKMeansIntegration:
    """Integration tests for K-Means"""

    def test_kmeans_end_to_end(self, kmeans_data):
        """Test complete K-Means pipeline"""
        if kmeans_data is None:
            pytest.skip("Data file not available")
        X = kmeans_data['X']
        K = 3
        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert centroids.shape == (K, X.shape[1])
        assert idx.shape[0] == X.shape[0]

    def test_kmeans_on_synthetic_data(self, synthetic_clustering_data):
        """Test K-Means on synthetic clustering data"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=20)
        unique_clusters = len(np.unique(idx))
        assert unique_clusters == K, f"Should have {K} clusters, got {unique_clusters}"


class TestSklearnComparison:
    """Tests comparing our implementation with scikit-learn"""

    def test_sklearn_kmeans(self, kmeans_data):
        """Test scikit-learn K-Means on same data"""
        if kmeans_data is None:
            pytest.skip("Data file not available")
        from sklearn.cluster import KMeans
        X = kmeans_data['X']
        model = KMeans(n_clusters=3, random_state=42, n_init=10)
        model.fit(X)
        centroids = model.cluster_centers_
        assert centroids.shape == (3, X.shape[1]), "Sklearn should produce correct shape"

    def test_comparison_with_sklearn(self, synthetic_clustering_data):
        """Compare our implementation with scikit-learn"""
        from sklearn.cluster import KMeans
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        np.random.seed(42)
        initial_centroids = kMeansInitCentroids(X, K)
        our_centroids, our_idx = runKMeans(X, initial_centroids, max_iters=50)
        sklearn_model = KMeans(n_clusters=K, init=initial_centroids, n_init=1, random_state=42)
        sklearn_model.fit(X)
        sklearn_centroids = sklearn_model.cluster_centers_
        assert our_centroids.shape == sklearn_centroids.shape, "Should have same shape"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample(self):
        """Test with single sample"""
        X = np.array([[1, 2]])
        K = 1
        centroids = kMeansInitCentroids(X, K)
        idx = findClosestCentroids(X, centroids)
        assert idx[0] == 0, "Single sample should be in cluster 0"

    def test_single_feature(self):
        """Test with single feature"""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert centroids.shape == (K, 1), "Should work with single feature"

    def test_k_equals_n(self):
        """Test with K equal to number of samples"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert len(np.unique(idx)) == K, "Each point should be its own cluster"

    def test_large_k(self, synthetic_clustering_data):
        """Test with large K relative to data size"""
        X = synthetic_clustering_data['X'][:10]
        K = 5
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=10)
        assert centroids.shape[0] == K, "Should handle large K"

    def test_zero_iterations(self, synthetic_clustering_data):
        """Test with zero iterations"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=0)
        np.testing.assert_array_equal(centroids, initial_centroids)

    def test_all_same_points(self):
        """Test with all identical points"""
        X = np.ones((10, 2))
        K = 2
        centroids = np.array([[1, 1], [1, 1]])
        idx = findClosestCentroids(X, centroids)
        assert len(idx) == 10, "Should handle identical points"


class TestClusterQuality:
    """Tests for cluster quality metrics"""

    def test_within_cluster_variance(self, synthetic_clustering_data):
        """Test that within-cluster variance decreases with iterations"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        np.random.seed(42)
        initial_centroids = kMeansInitCentroids(X, K)
        centroids1, idx1 = runKMeans(X, initial_centroids, max_iters=1)
        centroids2, idx2 = runKMeans(X, initial_centroids, max_iters=10)
        def compute_variance(X, centroids, idx):
            variance = 0
            for i in range(len(centroids)):
                cluster_points = X[idx == i]
                if len(cluster_points) > 0:
                    variance += np.sum((cluster_points - centroids[i]) ** 2)
            return variance
        var1 = compute_variance(X, centroids1, idx1)
        var2 = compute_variance(X, centroids2, idx2)
        assert var2 <= var1, "Variance should not increase with more iterations"

    def test_cluster_sizes(self, synthetic_clustering_data):
        """Test that clusters have reasonable sizes"""
        X = synthetic_clustering_data['X']
        K = synthetic_clustering_data['n_clusters']
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, max_iters=20)
        cluster_sizes = [np.sum(idx == i) for i in range(K)]
        assert all(size > 0 for size in cluster_sizes), "All clusters should have at least one point"
