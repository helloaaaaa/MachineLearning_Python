# -*- coding: utf-8 -*-
"""
K-Means聚类算法测试模块
测试内容：
- 数据加载功能
- 最近类中心查找
- 类中心计算
- K-Means迭代过程
- 类中心初始化
- 端到端集成测试
- 与 scikit-learn 对比
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 动态导入模块
import importlib.util
spec = importlib.util.spec_from_file_location("KMeans", 
    os.path.join(PROJECT_ROOT, "K-Means", "K-Menas.py"))
KMeans = importlib.util.module_from_spec(spec)
spec.loader.exec_module(KMeans)

findClosestCentroids = KMeans.findClosestCentroids
computerCentroids = KMeans.computerCentroids
runKMeans = KMeans.runKMeans
kMeansInitCentroids = KMeans.kMeansInitCentroids

from scipy import io as spio


try:
    from sklearn.cluster import KMeans as SklearnKMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestDataLoading:
    """数据加载测试类"""

    def test_load_mat_data(self, kmeans_data_path):
        """测试 mat 文件加载"""
        if os.path.exists(kmeans_data_path):
            data = spio.loadmat(kmeans_data_path)
            assert isinstance(data, dict)
            assert 'X' in data
            X = data['X']
            assert isinstance(X, np.ndarray)
            assert X.ndim == 2


class TestFindClosestCentroids:
    """最近类中心查找测试类"""

    def test_find_closest_centroids_shape(self, sample_clustering_data):
        """测试输出形状"""
        X, K, centroids = sample_clustering_data
        idx = findClosestCentroids(X, centroids)
        assert idx.shape == (X.shape[0],)

    def test_find_closest_centroids_range(self, sample_clustering_data):
        """测试输出范围"""
        X, K, centroids = sample_clustering_data
        idx = findClosestCentroids(X, centroids)
        assert np.all(idx >= 0)
        assert np.all(idx < K)

    def test_find_closest_centroids_exact(self):
        """测试精确查找"""
        X = np.array([[0, 0], [10, 10], [0, 10]])
        centroids = np.array([[0, 0], [10, 10]])
        idx = findClosestCentroids(X, centroids)
        # 第一个点应属于类0，第二个属于类1，第三个与类0更近
        expected = np.array([0, 1, 0])
        np.testing.assert_array_equal(idx, expected)

    def test_find_closest_centroids_single_point(self):
        """测试单点"""
        X = np.array([[5, 5]])
        centroids = np.array([[0, 0], [10, 10]])
        idx = findClosestCentroids(X, centroids)
        # 与两个类中心等距，可以属于任意一类
        assert idx[0] in [0, 1]


class TestComputeCentroids:
    """类中心计算测试类"""

    def test_compute_centroids_shape(self, sample_clustering_data):
        """测试输出形状"""
        X, K, initial_centroids = sample_clustering_data
        # 先分配每个点到最近的类中心
        idx = findClosestCentroids(X, initial_centroids)
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (K, X.shape[1])

    def test_compute_centroids_exact(self):
        """测试精确计算"""
        X = np.array([[0, 0], [0, 1], [10, 10], [10, 11]])
        idx = np.array([0, 0, 1, 1])
        K = 2
        centroids = computerCentroids(X, idx, K)
        # 类0中心应为 [0, 0.5]，类1中心应为 [10, 10.5]
        expected = np.array([[0, 0.5], [10, 10.5]])
        np.testing.assert_array_almost_equal(centroids, expected)

    def test_compute_centroids_empty_cluster(self):
        """测试空类情况"""
        X = np.array([[0, 0], [1, 1]])
        idx = np.array([0, 0])
        K = 2
        # 类1没有样本，中心应为 [0, 0]
        centroids = computerCentroids(X, idx, K)
        assert centroids.shape == (2, 2)
        # 类0中心应为 [0.5, 0.5]
        np.testing.assert_array_almost_equal(centroids[0], [0.5, 0.5])


class TestKMeansInitCentroids:
    """类中心初始化测试类"""

    def test_init_centroids_shape(self, sample_clustering_data):
        """测试初始化形状"""
        X, K, _ = sample_clustering_data
        centroids = kMeansInitCentroids(X, K)
        assert centroids.shape == (K, X.shape[1])

    def test_init_centroids_from_data(self, sample_clustering_data):
        """测试初始化点来自数据"""
        X, K, _ = sample_clustering_data
        centroids = kMeansInitCentroids(X, K)
        # 每个中心应存在于原始数据中
        for centroid in centroids:
            matches = np.all(np.isclose(X, centroid), axis=1)
            assert np.any(matches)

    def test_init_centroids_randomness(self, sample_clustering_data):
        """测试初始化随机性"""
        X, K, _ = sample_clustering_data
        centroids1 = kMeansInitCentroids(X, K)
        centroids2 = kMeansInitCentroids(X, K)
        # 两次初始化很可能不同
        assert not np.allclose(centroids1, centroids2)


class TestRunKMeans:
    """K-Means运行测试类"""

    def test_run_kmeans_shape(self, sample_clustering_data):
        """测试输出形状"""
        X, K, initial_centroids = sample_clustering_data
        max_iters = 10
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        assert centroids.shape == (K, X.shape[1])
        assert idx.shape == (X.shape[0],)

    def test_run_kmeans_convergence(self, sample_clustering_data):
        """测试收敛性"""
        X, K, _ = sample_clustering_data
        initial_centroids = kMeansInitCentroids(X, K)
        max_iters = 100
        centroids, idx = runKMeans(X, initial_centroids, max_iters, False)
        # 验证每个点都被分配
        assert len(idx) == X.shape[0]
        assert np.all(idx >= 0)
        assert np.all(idx < K)

    def test_run_kmeans_improvement(self, sample_clustering_data):
        """测试聚类改进"""
        X, K, _ = sample_clustering_data
        initial_centroids = kMeansInitCentroids(X, K)
        
        # 计算初始分配的距离平方和
        idx_initial = findClosestCentroids(X, initial_centroids)
        initial_cost = np.sum([
            np.sum((X[i] - initial_centroids[idx_initial[i]]) ** 2)
            for i in range(X.shape[0])
        ])
        
        # 运行K-Means
        centroids_final, idx_final = runKMeans(X, initial_centroids, 50, False)
        final_cost = np.sum([
            np.sum((X[i] - centroids_final[idx_final[i]]) ** 2)
            for i in range(X.shape[0])
        ])
        
        # 最终代价应小于等于初始代价
        assert final_cost <= initial_cost * 1.01  # 允许1%误差


class TestIntegration:
    """集成测试类"""

    def test_kmeans_end_to_end(self, sample_clustering_data):
        """端到端集成测试"""
        X, K, _ = sample_clustering_data
        
        # 初始化
        initial_centroids = kMeansInitCentroids(X, K)
        
        # 运行K-Means
        centroids, idx = runKMeans(X, initial_centroids, 100, False)
        
        # 验证结果
        assert centroids.shape == (K, X.shape[1])
        assert idx.shape == (X.shape[0],)
        assert np.all(idx >= 0)
        assert np.all(idx < K)
        
        # 计算最终代价
        final_cost = np.sum([
            np.sum((X[i] - centroids[idx[i]]) ** 2)
            for i in range(X.shape[0])
        ])
        assert final_cost >= 0

    def test_kmeans_on_well_separated_clusters(self):
        """测试在明显分离的数据上的聚类"""
        np.random.seed(42)
        # 创建三个明显分离的簇
        cluster1 = np.random.randn(30, 2) + np.array([0, 0])
        cluster2 = np.random.randn(30, 2) + np.array([10, 0])
        cluster3 = np.random.randn(30, 2) + np.array([5, 10])
        X = np.vstack([cluster1, cluster2, cluster3])
        
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, 100, False)
        
        # 每个簇应主要包含来自同一真实簇的点
        for k in range(K):
            cluster_points = idx[k*30:(k+1)*30]
            # 找到该段中最常见的类别
            unique, counts = np.unique(cluster_points, return_counts=True)
            most_common_count = np.max(counts)
            # 至少60%的点应属于同一类别
            assert most_common_count >= 18


class TestEdgeCases:
    """边界情况测试类"""

    def test_single_cluster(self):
        """测试K=1"""
        X = np.random.randn(50, 2)
        K = 1
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, 10, False)
        assert centroids.shape == (1, 2)
        assert np.all(idx == 0)

    def test_more_clusters_than_points(self):
        """测试K大于样本数"""
        X = np.random.randn(5, 2)
        K = 10
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, 10, False)
        # 只有部分类中心会被使用
        unique_idx = np.unique(idx)
        assert len(unique_idx) <= X.shape[0]

    def test_high_dimensional_data(self):
        """测试高维数据"""
        np.random.seed(42)
        X = np.random.randn(100, 50)
        K = 3
        initial_centroids = kMeansInitCentroids(X, K)
        centroids, idx = runKMeans(X, initial_centroids, 20, False)
        assert centroids.shape == (K, 50)
        assert idx.shape == (100,)

    def test_zero_iterations(self):
        """测试0次迭代"""
        X, K, initial_centroids = sample_clustering_data
        centroids, idx = runKMeans(X, initial_centroids, 0, False)
        # 类中心应保持不变
        np.testing.assert_array_equal(centroids, initial_centroids)


class TestSklearnComparison:
    """与 scikit-learn 对比测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_compare_with_sklearn(self, sample_clustering_data):
        """与 sklearn K-Means 对比"""
        X, K, _ = sample_clustering_data
        
        # 自定义实现
        initial_centroids = kMeansInitCentroids(X, K)
        centroids_custom, idx_custom = runKMeans(X, initial_centroids, 100, False)
        cost_custom = np.sum([
            np.sum((X[i] - centroids_custom[idx_custom[i]]) ** 2)
            for i in range(X.shape[0])
        ])
        
        # sklearn 实现
        kmeans = SklearnKMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(X)
        cost_sklearn = kmeans.inertia_
        
        # 两者代价应相近（允许一定差异）
        ratio = cost_custom / cost_sklearn if cost_sklearn > 0 else 1
        assert 0.5 <= ratio <= 2.0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_predict_consistency(self, sample_clustering_data):
        """测试预测一致性"""
        X, K, _ = sample_clustering_data
        
        # 自定义实现
        initial_centroids = kMeansInitCentroids(X, K)
        centroids_custom, _ = runKMeans(X, initial_centroids, 50, False)
        
        # sklearn 实现
        kmeans = SklearnKMeans(n_clusters=K, random_state=42, n_init=10)
        kmeans.fit(X)
        
        # 比较类中心（可能需要重新排序）
        # 由于K-Means可能收敛到不同局部最优，我们只比较代价
        cost_custom = np.sum(np.min([
            np.sum((X - c) ** 2, axis=1) for c in centroids_custom
        ], axis=0))
        
        ratio = cost_custom / kmeans.inertia_ if kmeans.inertia_ > 0 else 1
        assert 0.5 <= ratio <= 2.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
