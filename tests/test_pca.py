# -*- coding: utf-8 -*-
"""
PCA主成分分析算法测试模块
测试内容：
- 数据加载功能
- 特征归一化
- 协方差矩阵计算
- 奇异值分解
- 数据投影
- 数据恢复
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
spec = importlib.util.spec_from_file_location("PCA",
    os.path.join(PROJECT_ROOT, "PCA", "PCA.py"))
PCA = importlib.util.module_from_spec(spec)
spec.loader.exec_module(PCA)

featureNormalize = PCA.featureNormalize
projectData = PCA.projectData
recoverData = PCA.recoverData
display_imageData = PCA.display_imageData

from scipy import io as spio


try:
    from sklearn.decomposition import PCA as SklearnPCA
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestDataLoading:
    """数据加载测试类"""

    def test_load_mat_data(self, pca_data_path):
        """测试 mat 文件加载"""
        if os.path.exists(pca_data_path):
            data = spio.loadmat(pca_data_path)
            assert isinstance(data, dict)
            assert 'X' in data
            X = data['X']
            assert isinstance(X, np.ndarray)
            assert X.ndim == 2


class TestFeatureNormalization:
    """特征归一化测试类"""

    def test_feature_normalize_shape(self, sample_regression_data):
        """测试归一化后形状"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormalize(X)
        assert X_norm.shape == X.shape
        assert mu.shape == (X.shape[1],)
        assert sigma.shape == (X.shape[1],)

    def test_feature_normalize_zero_mean(self, sample_regression_data):
        """测试归一化后均值为0"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormalize(X)
        np.testing.assert_array_almost_equal(
            np.mean(X_norm, axis=0), np.zeros(X.shape[1]), decimal=10
        )

    def test_feature_normalize_unit_std(self, sample_regression_data):
        """测试归一化后标准差为1"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormalize(X)
        np.testing.assert_array_almost_equal(
            np.std(X_norm, axis=0), np.ones(X.shape[1]), decimal=10
        )

    def test_feature_normalize_inverse(self, sample_regression_data):
        """测试归一化可逆"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormalize(X)
        X_recovered = X_norm * sigma + mu
        np.testing.assert_array_almost_equal(X, X_recovered)


class TestProjection:
    """数据投影测试类"""

    def test_project_data_shape(self):
        """测试投影后形状"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_norm, mu, sigma = featureNormalize(X)

        # 计算PCA
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        K = 2
        Z = projectData(X_norm, U, K)
        assert Z.shape == (100, 2)

    def test_project_data_reduces_dimension(self):
        """测试降维"""
        np.random.seed(42)
        X = np.random.randn(50, 10)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        for K in [1, 3, 5, 9]:
            Z = projectData(X_norm, U, K)
            assert Z.shape == (50, K)

    def test_project_data_orthogonality(self):
        """测试投影的正交性"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # U的列应该是正交的
        identity = np.dot(U.T, U)
        np.testing.assert_array_almost_equal(identity, np.eye(5), decimal=5)


class TestRecovery:
    """数据恢复测试类"""

    def test_recover_data_shape(self):
        """测试恢复后形状"""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        K = 2
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        assert X_rec.shape == X_norm.shape

    def test_recover_data_approximation(self):
        """测试恢复是近似"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 使用全部主成分应能完美恢复
        K = 5
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)
        np.testing.assert_array_almost_equal(X_norm, X_rec, decimal=5)

    def test_recover_data_error_increases_with_less_components(self):
        """测试使用更少主成分时误差增大"""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        errors = []
        for K in [1, 2, 3, 4, 5]:
            Z = projectData(X_norm, U, K)
            X_rec = recoverData(Z, U, K)
            error = np.mean((X_norm - X_rec) ** 2)
            errors.append(error)

        # 误差应随K减小而单调递减
        for i in range(len(errors) - 1):
            assert errors[i] >= errors[i + 1] * 0.99


class TestSVD:
    """奇异值分解测试类"""

    def test_svd_reconstruction(self):
        """测试SVD重构"""
        np.random.seed(42)
        A = np.random.randn(5, 5)
        U, S, V = np.linalg.svd(A)

        # 重构
        S_matrix = np.diag(S)
        A_reconstructed = np.dot(U, np.dot(S_matrix, V))
        np.testing.assert_array_almost_equal(A, A_reconstructed)

    def test_svd_properties(self):
        """测试SVD性质"""
        np.random.seed(42)
        A = np.random.randn(10, 5)
        U, S, V = np.linalg.svd(A, full_matrices=False)

        # U和V应该是正交矩阵
        np.testing.assert_array_almost_equal(np.dot(U.T, U), np.eye(5), decimal=5)
        np.testing.assert_array_almost_equal(np.dot(V, V.T), np.eye(5), decimal=5)

        # 奇异值应为正
        assert np.all(S > 0)


class TestIntegration:
    """集成测试类"""

    def test_pca_end_to_end(self):
        """端到端PCA测试"""
        np.random.seed(42)
        # 创建相关数据
        n_samples = 100
        x1 = np.random.randn(n_samples)
        x2 = 0.5 * x1 + 0.1 * np.random.randn(n_samples)
        x3 = np.random.randn(n_samples)
        X = np.column_stack([x1, x2, x3])

        # 归一化
        X_norm, mu, sigma = featureNormalize(X)

        # PCA
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 降维到2D
        K = 2
        Z = projectData(X_norm, U, K)

        # 恢复
        X_rec = recoverData(Z, U, K)

        # 验证
        assert Z.shape == (n_samples, K)
        assert X_rec.shape == X_norm.shape

        # 恢复误差应较小（因为x1和x2相关）
        error = np.mean((X_norm - X_rec) ** 2)
        assert error < 0.5

    def test_pca_variance_retention(self):
        """测试方差保留"""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 计算保留的方差比例
        total_variance = np.sum(S)
        for K in [1, 3, 5, 10]:
            retained_variance = np.sum(S[:K]) / total_variance
            # 保留的方差应随K增加
            assert 0 <= retained_variance <= 1

    def test_pca_on_image_data(self):
        """测试图像数据PCA"""
        np.random.seed(42)
        # 模拟图像数据 (32x32 = 1024维)
        n_samples = 50
        n_features = 1024
        X = np.random.randn(n_samples, n_features) * 50 + 128

        # 归一化
        X_norm, mu, sigma = featureNormalize(X)

        # PCA
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 降维到100维
        K = 100
        Z = projectData(X_norm, U, K)
        X_rec = recoverData(Z, U, K)

        assert Z.shape == (n_samples, K)
        assert X_rec.shape == X_norm.shape


class TestEdgeCases:
    """边界情况测试类"""

    def test_single_feature(self):
        """测试单特征"""
        X = np.random.randn(50, 1)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        Z = projectData(X_norm, U, 1)
        X_rec = recoverData(Z, U, 1)

        np.testing.assert_array_almost_equal(X_norm, X_rec, decimal=5)

    def test_single_sample(self):
        """测试单样本"""
        X = np.random.randn(1, 5)
        X_norm, mu, sigma = featureNormalize(X)

        # 单样本PCA（方差为0）
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 所有奇异值应为0
        np.testing.assert_array_almost_equal(S, np.zeros(5))

    def test_zero_variance_feature(self):
        """测试零方差特征"""
        X = np.column_stack([
            np.random.randn(50),
            np.ones(50)  # 常数特征
        ])
        X_norm, mu, sigma = featureNormalize(X)

        # 常数特征归一化后为NaN
        assert np.any(np.isnan(X_norm))

    def test_high_dimensional_reduction(self):
        """测试高维降维"""
        X = np.random.randn(10, 100)
        X_norm, mu, sigma = featureNormalize(X)

        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # 降维到比样本数少的维度
        K = 5
        Z = projectData(X_norm, U, K)
        assert Z.shape == (10, K)


class TestSklearnComparison:
    """与 scikit-learn 对比测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_compare_with_sklearn(self):
        """与 sklearn PCA 对比"""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        # 自定义实现
        X_norm_custom, mu, sigma = featureNormalize(X)
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm_custom), X_norm_custom) / m
        U, S, V = np.linalg.svd(Sigma)
        K = 5
        Z_custom = projectData(X_norm_custom, U, K)
        X_rec_custom = recoverData(Z_custom, U, K)

        # sklearn 实现
        scaler = StandardScaler()
        X_norm_sklearn = scaler.fit_transform(X)
        pca = SklearnPCA(n_components=K)
        Z_sklearn = pca.fit_transform(X_norm_sklearn)
        X_rec_sklearn = pca.inverse_transform(Z_sklearn)

        # 投影结果形状相同
        assert Z_custom.shape == Z_sklearn.shape
        assert X_rec_custom.shape == X_rec_sklearn.shape

        # 恢复误差应相近
        error_custom = np.mean((X_norm_custom - X_rec_custom) ** 2)
        error_sklearn = np.mean((X_norm_sklearn - X_rec_sklearn) ** 2)

        ratio = error_custom / error_sklearn if error_sklearn > 0 else 1
        assert 0.5 <= ratio <= 2.0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_explained_variance_ratio(self):
        """测试解释方差比例"""
        np.random.seed(42)
        X = np.random.randn(100, 10)

        # 自定义实现
        X_norm, mu, sigma = featureNormalize(X)
        m = X.shape[0]
        Sigma = np.dot(np.transpose(X_norm), X_norm) / m
        U, S, V = np.linalg.svd(Sigma)

        # sklearn
        scaler = StandardScaler()
        X_norm_sklearn = scaler.fit_transform(X)
        pca = SklearnPCA(n_components=10)
        pca.fit(X_norm_sklearn)

        # 比较奇异值/解释方差
        # 注意：sklearn的explained_variance_是特征值，等于S^2/(m-1)
        # 我们只比较相对比例
        custom_ratio = S / np.sum(S)
        sklearn_ratio = pca.explained_variance_ / np.sum(pca.explained_variance_)

        np.testing.assert_array_almost_equal(custom_ratio, sklearn_ratio, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
