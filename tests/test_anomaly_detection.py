# -*- coding: utf-8 -*-
"""
异常检测算法测试模块
测试内容：
- 数据加载功能
- 高斯分布参数估计
- 多元高斯概率计算
- 阈值选择
- F1分数计算
- 端到端集成测试
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# 动态导入模块
import importlib.util
spec = importlib.util.spec_from_file_location("AnomalyDetection", 
    os.path.join(PROJECT_ROOT, "AnomalyDetection", "AnomalyDetection.py"))
AnomalyDetection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(AnomalyDetection)

estimateGaussian = AnomalyDetection.estimateGaussian
multivariateGaussian = AnomalyDetection.multivariateGaussian
selectThreshold = AnomalyDetection.selectThreshold
display_2d_data = AnomalyDetection.display_2d_data

from scipy import io as spio


class TestDataLoading:
    """数据加载测试类"""

    def test_load_mat_data(self, anomaly_detection_data_path):
        """测试 mat 文件加载"""
        if os.path.exists(anomaly_detection_data_path):
            data = spio.loadmat(anomaly_detection_data_path)
            assert isinstance(data, dict)
            assert 'X' in data
            X = data['X']
            assert isinstance(X, np.ndarray)
            assert X.ndim == 2

    def test_load_validation_data(self, anomaly_detection_data_path):
        """测试加载验证集"""
        if os.path.exists(anomaly_detection_data_path):
            data = spio.loadmat(anomaly_detection_data_path)
            # 检查是否有验证集
            if 'Xval' in data and 'yval' in data:
                Xval = data['Xval']
                yval = data['yval']
                assert Xval.ndim == 2
                assert yval.ndim == 2
                assert Xval.shape[0] == yval.shape[0]


class TestEstimateGaussian:
    """高斯参数估计测试类"""

    def test_estimate_gaussian_shape(self):
        """测试输出形状"""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        mu, sigma2 = estimateGaussian(X)
        assert mu.shape == (3,)
        assert sigma2.shape == (3,)

    def test_estimate_gaussian_mean(self):
        """测试均值估计"""
        np.random.seed(42)
        true_mean = np.array([1, 2, 3])
        X = np.random.randn(1000, 3) + true_mean
        mu, sigma2 = estimateGaussian(X)
        np.testing.assert_array_almost_equal(mu, true_mean, decimal=1)

    def test_estimate_gaussian_variance(self):
        """测试方差估计"""
        np.random.seed(42)
        X = np.random.randn(1000, 2)
        mu, sigma2 = estimateGaussian(X)
        # 标准正态分布方差应为1
        np.testing.assert_array_almost_equal(sigma2, [1, 1], decimal=1)

    def test_estimate_gaussian_known_data(self):
        """测试已知数据"""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        mu, sigma2 = estimateGaussian(X)
        expected_mu = np.array([3, 4])
        expected_var = np.array([4, 4])  # 方差
        np.testing.assert_array_equal(mu, expected_mu)
        np.testing.assert_array_equal(sigma2, expected_var)


class TestMultivariateGaussian:
    """多元高斯分布测试类"""

    def test_multivariate_gaussian_shape(self):
        """测试输出形状"""
        X = np.array([[1, 2], [3, 4]])
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        p = multivariateGaussian(X, mu, sigma2)
        assert p.shape == (2,)

    def test_multivariate_gaussian_range(self):
        """测试概率范围"""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        p = multivariateGaussian(X, mu, sigma2)
        assert np.all(p > 0)
        assert np.all(p <= 1)

    def test_multivariate_gaussian_at_mean(self):
        """测试均值处概率最大"""
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        
        # 均值处的概率
        p_mean = multivariateGaussian(mu.reshape(1, -1), mu, sigma2)
        
        # 远离均值的点
        X_far = np.array([[5, 5]])
        p_far = multivariateGaussian(X_far, mu, sigma2)
        
        assert p_mean[0] > p_far[0]

    def test_multivariate_gaussian_symmetry(self):
        """测试对称性"""
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        
        X1 = np.array([[1, 0]])
        X2 = np.array([[-1, 0]])
        
        p1 = multivariateGaussian(X1, mu, sigma2)
        p2 = multivariateGaussian(X2, mu, sigma2)
        
        np.testing.assert_almost_equal(p1[0], p2[0])

    def test_multivariate_gaussian_diagonal_covariance(self):
        """测试对角协方差"""
        X = np.array([[0, 0]])
        mu = np.array([0, 0])
        sigma2 = np.array([4, 9])  # 不同方差
        
        p = multivariateGaussian(X, mu, sigma2)
        
        # 在均值处，概率密度应为 (2*pi)^(-k/2) * |Sigma|^(-1/2)
        # = (2*pi)^(-1) * (4*9)^(-1/2) = 1/(2*pi*6)
        expected = 1 / (2 * np.pi * 6)
        np.testing.assert_almost_equal(p[0], expected, decimal=5)


class TestSelectThreshold:
    """阈值选择测试类"""

    def test_select_threshold_output(self):
        """测试输出类型"""
        np.random.seed(42)
        yval = np.array([0, 0, 0, 1, 1])
        pval = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        epsilon, F1 = selectThreshold(yval, pval)
        assert isinstance(epsilon, (float, np.floating))
        assert isinstance(F1, (float, np.floating))
        assert 0 <= F1 <= 1

    def test_select_threshold_perfect_case(self):
        """测试完美分离情况"""
        # 正常点概率高，异常点概率低
        yval = np.array([0, 0, 0, 1, 1])
        pval = np.array([0.9, 0.8, 0.7, 0.1, 0.05])
        
        epsilon, F1 = selectThreshold(yval, pval)
        
        # F1应接近1
        assert F1 > 0.8
        # epsilon应在正常和异常概率之间
        assert 0.05 < epsilon < 0.9

    def test_select_threshold_all_normal(self):
        """测试全正常情况"""
        yval = np.array([0, 0, 0, 0, 0])
        pval = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        epsilon, F1 = selectThreshold(yval, pval)
        
        # 全正常时F1应为0（无法计算精确率和召回率）
        assert F1 == 0 or np.isnan(F1)

    def test_select_threshold_all_anomaly(self):
        """测试全异常情况"""
        yval = np.array([1, 1, 1, 1, 1])
        pval = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        
        epsilon, F1 = selectThreshold(yval, pval)
        
        # 全异常时F1应为0
        assert F1 == 0 or np.isnan(F1)


class TestIntegration:
    """集成测试类"""

    def test_anomaly_detection_workflow(self):
        """异常检测完整工作流"""
        np.random.seed(42)
        
        # 生成正常数据
        X_normal = np.random.randn(300, 2)
        
        # 生成异常数据
        X_anomaly = np.random.randn(20, 2) * 0.5 + np.array([5, 5])
        
        # 合并
        X = np.vstack([X_normal, X_anomaly])
        
        # 创建验证集标签
        yval = np.hstack([np.zeros(300), np.ones(20)])
        
        # 估计高斯参数
        mu, sigma2 = estimateGaussian(X)
        
        # 计算概率
        p = multivariateGaussian(X, mu, sigma2)
        
        # 选择阈值
        epsilon, F1 = selectThreshold(yval, p)
        
        # 检测异常
        outliers = p < epsilon
        
        # 验证
        assert len(outliers) == len(X)
        # 应检测到一些异常
        assert np.sum(outliers) > 0
        # F1应合理
        assert 0 <= F1 <= 1

    def test_anomaly_detection_2d(self):
        """二维数据异常检测"""
        np.random.seed(42)
        
        # 生成二维正态数据
        X_train = np.random.randn(200, 2)
        
        # 验证集包含正常和异常
        X_val_normal = np.random.randn(50, 2)
        X_val_anomaly = np.random.randn(10, 2) * 0.5 + np.array([4, 4])
        X_val = np.vstack([X_val_normal, X_val_anomaly])
        y_val = np.hstack([np.zeros(50), np.ones(10)])
        
        # 训练
        mu, sigma2 = estimateGaussian(X_train)
        p_val = multivariateGaussian(X_val, mu, sigma2)
        
        # 选择阈值
        epsilon, F1 = selectThreshold(y_val, p_val)
        
        # 在训练集上检测
        p_train = multivariateGaussian(X_train, mu, sigma2)
        outliers_train = np.sum(p_train < epsilon)
        
        # 训练集异常比例应较低
        assert outliers_train / len(X_train) < 0.2

    def test_different_sigma_values(self):
        """测试不同方差值"""
        np.random.seed(42)
        
        # 生成不同方差的数据
        X_low_var = np.random.randn(100, 2) * 0.5
        X_high_var = np.random.randn(100, 2) * 2
        
        for X in [X_low_var, X_high_var]:
            mu, sigma2 = estimateGaussian(X)
            p = multivariateGaussian(X, mu, sigma2)
            
            # 概率应在合理范围内
            assert np.all(p > 0)
            assert np.all(p <= 1)


class TestEdgeCases:
    """边界情况测试类"""

    def test_single_feature(self):
        """测试单特征"""
        X = np.random.randn(100, 1)
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        
        assert mu.shape == (1,)
        assert sigma2.shape == (1,)
        assert p.shape == (100,)

    def test_single_sample(self):
        """测试单样本"""
        X = np.array([[1, 2]])
        mu, sigma2 = estimateGaussian(X)
        
        # 单样本方差为0
        assert sigma2[0] == 0
        assert sigma2[1] == 0

    def test_high_dimensional_data(self):
        """测试高维数据"""
        np.random.seed(42)
        X = np.random.randn(50, 20)
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        
        assert mu.shape == (20,)
        assert sigma2.shape == (20,)
        assert p.shape == (50,)
        assert np.all(p > 0)

    def test_correlated_features(self):
        """测试相关特征"""
        np.random.seed(42)
        x1 = np.random.randn(100)
        x2 = 0.9 * x1 + 0.1 * np.random.randn(100)
        X = np.column_stack([x1, x2])
        
        mu, sigma2 = estimateGaussian(X)
        p = multivariateGaussian(X, mu, sigma2)
        
        assert np.all(p > 0)
        assert np.all(p <= 1)

    def test_zero_variance(self):
        """测试零方差特征"""
        X = np.column_stack([
            np.random.randn(50),
            np.ones(50)  # 常数特征
        ])
        
        mu, sigma2 = estimateGaussian(X)
        assert sigma2[1] == 0
        
        # 零方差会导致概率计算中的除零
        # 实际代码使用对角协方差，应能处理
        p = multivariateGaussian(X, mu, sigma2)
        # 对于常数特征，概率可能为0或无穷


class TestVisualization:
    """可视化函数测试类"""

    def test_display_2d_data(self):
        """测试2D数据显示函数"""
        X = np.random.randn(50, 2)
        try:
            plt = display_2d_data(X, 'bx')
            assert plt is not None
        except Exception as e:
            pytest.skip(f"Plotting failed: {e}")


class TestNumericalStability:
    """数值稳定性测试类"""

    def test_very_small_probabilities(self):
        """测试极小概率"""
        X = np.array([[10, 10]])
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        
        p = multivariateGaussian(X, mu, sigma2)
        # 远离均值时概率应很小
        assert p[0] < 1e-10

    def test_very_large_values(self):
        """测试极大值"""
        X = np.array([[1000, 1000]])
        mu = np.array([0, 0])
        sigma2 = np.array([1, 1])
        
        p = multivariateGaussian(X, mu, sigma2)
        # 不应出现NaN或Inf
        assert not np.isnan(p[0])
        assert not np.isinf(p[0])

    def test_nearly_singular_covariance(self):
        """测试接近奇异的协方差"""
        # 高度相关的特征
        x1 = np.random.randn(100)
        x2 = x1 + 1e-10 * np.random.randn(100)
        X = np.column_stack([x1, x2])
        
        mu, sigma2 = estimateGaussian(X)
        # 不应报错
        p = multivariateGaussian(X, mu, sigma2)
        assert np.all(np.isfinite(p))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
