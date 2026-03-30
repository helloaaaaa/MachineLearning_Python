# -*- coding: utf-8 -*-
"""
线性回归算法测试模块
测试内容：
- 数据加载功能
- 特征归一化
- 代价函数计算
- 梯度下降算法
- 端到端集成测试
- 与 scikit-learn 对比
"""
import pytest
import numpy as np
import os
import sys

# 添加 LinearRegression 目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LinearRegression"))

from LinearRegression import (
    loadtxtAndcsv_data,
    loadnpy_data,
    featureNormaliza,
    computerCost,
    gradientDescent,
    linearRegression
)


class TestDataLoading:
    """数据加载测试类"""

    def test_loadtxt_and_csv_data(self, linear_regression_data_path):
        """测试 txt 和 csv 数据加载功能"""
        if os.path.exists(linear_regression_data_path):
            data = loadtxtAndcsv_data(linear_regression_data_path, ",", np.float64)
            assert isinstance(data, np.ndarray)
            assert data.ndim == 2
            assert data.shape[0] > 0
            assert data.shape[1] >= 2

    def test_loadtxt_nonexistent_file(self):
        """测试加载不存在的文件"""
        with pytest.raises((FileNotFoundError, OSError)):
            loadtxtAndcsv_data("nonexistent_file.txt", ",", np.float64)

    def test_loadnpy_data(self):
        """测试 npy 数据加载功能"""
        npy_path = os.path.join(PROJECT_ROOT, "LinearRegression", "data.npy")
        if os.path.exists(npy_path):
            data = loadnpy_data(npy_path)
            assert isinstance(data, np.ndarray)


class TestFeatureNormalization:
    """特征归一化测试类"""

    def test_feature_normaliza_shape(self, sample_regression_data):
        """测试归一化后数据形状保持不变"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape
        assert mu.shape == (X.shape[1],)
        assert sigma.shape == (X.shape[1],)

    def test_feature_normaliza_zero_mean_unit_std(self, sample_regression_data):
        """测试归一化后均值为0，标准差为1"""
        X, _, _ = sample_regression_data
        X_norm, mu, sigma = featureNormaliza(X)
        # 允许一定的数值误差
        np.testing.assert_array_almost_equal(np.mean(X_norm, axis=0), 
                                              np.zeros(X.shape[1]), decimal=10)
        np.testing.assert_array_almost_equal(np.std(X_norm, axis=0), 
                                              np.ones(X.shape[1]), decimal=10)

    def test_feature_normaliza_single_feature(self):
        """测试单特征归一化"""
        X = np.array([[1], [2], [3], [4], [5]], dtype=np.float64)
        X_norm, mu, sigma = featureNormaliza(X)
        assert X_norm.shape == X.shape
        np.testing.assert_almost_equal(np.mean(X_norm), 0, decimal=10)

    def test_feature_normaliza_constant_feature(self):
        """测试常数特征归一化（标准差为0的情况）"""
        X = np.array([[1, 5], [2, 5], [3, 5]], dtype=np.float64)
        X_norm, mu, sigma = featureNormaliza(X)
        # 常数特征归一化后应为 NaN 或 0
        assert X_norm.shape == X.shape


class TestCostFunction:
    """代价函数测试类"""

    def test_computer_cost_shape(self, sample_regression_data):
        """测试代价函数返回值类型"""
        X, y, true_theta = sample_regression_data
        m = len(y)
        X = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        J = computerCost(X, y, theta)
        assert isinstance(J, (float, np.floating, np.ndarray))

    def test_computer_cost_zero_theta(self, sample_regression_data):
        """测试 theta 为零时的代价"""
        X, y, _ = sample_regression_data
        m = len(y)
        X = np.hstack((np.ones((m, 1)), X))
        theta = np.zeros((X.shape[1], 1))
        y = y.reshape(-1, 1)
        J = computerCost(X, y, theta)
        # 代价应为正值
        assert J >= 0

    def test_computer_cost_perfect_fit(self):
        """测试完美拟合时的代价"""
        X = np.array([[1, 2], [1, 3], [1, 4]], dtype=np.float64)
        y = np.array([[5], [7], [9]], dtype=np.float64)
        theta = np.array([[1], [2]], dtype=np.float64)
        J = computerCost(X, y, theta)
        # 完美拟合时代价应接近0
        np.testing.assert_almost_equal(J, 0, decimal=5)


class TestGradientDescent:
    """梯度下降算法测试类"""

    def test_gradient_descent_convergence(self, sample_regression_data):
        """测试梯度下降是否收敛"""
        X, y, _ = sample_regression_data
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_norm = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_norm.shape[1], 1))
        y = y.reshape(-1, 1)
        
        alpha = 0.01
        num_iters = 100
        theta_final, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
        
        assert theta_final.shape == theta.shape
        assert len(J_history) == num_iters
        # 代价应随迭代递减
        assert J_history[-1] <= J_history[0]

    def test_gradient_descent_decreasing_cost(self, sample_regression_data):
        """测试代价函数单调递减"""
        X, y, _ = sample_regression_data
        m = len(y)
        X_norm, mu, sigma = featureNormaliza(X)
        X_norm = np.hstack((np.ones((m, 1)), X_norm))
        theta = np.zeros((X_norm.shape[1], 1))
        y = y.reshape(-1, 1)
        
        alpha = 0.01
        num_iters = 50
        _, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
        
        # 检查代价是否总体递减（允许微小波动）
        for i in range(1, len(J_history)):
            assert J_history[i] <= J_history[i-1] * 1.01  # 允许1%的误差


class TestIntegration:
    """集成测试类"""

    def test_linear_regression_end_to_end(self, sample_regression_data):
        """端到端集成测试"""
        X, y, true_theta = sample_regression_data
        m = len(y)
        
        # 数据归一化
        X_norm, mu, sigma = featureNormaliza(X)
        X_norm = np.hstack((np.ones((m, 1)), X_norm))
        y = y.reshape(-1, 1)
        
        # 训练模型
        theta = np.zeros((X_norm.shape[1], 1))
        alpha = 0.01
        num_iters = 200
        theta_final, J_history = gradientDescent(X_norm, y, theta, alpha, num_iters)
        
        # 验证结果
        assert theta_final.shape == (X_norm.shape[1], 1)
        assert J_history[-1] < J_history[0]
        
        # 预测验证
        predictions = np.dot(X_norm, theta_final)
        assert predictions.shape == y.shape

    def test_prediction_accuracy(self):
        """测试预测准确性"""
        # 创建简单线性数据 y = 2 + 3*x
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]], dtype=np.float64)
        y = np.array([[5], [8], [11], [14], [17]], dtype=np.float64)
        
        theta = np.zeros((2, 1))
        alpha = 0.1
        num_iters = 1000
        theta_final, _ = gradientDescent(X, y, theta, alpha, num_iters)
        
        # 验证学习到的参数接近真实值 [2, 3]
        np.testing.assert_array_almost_equal(
            theta_final.flatten(), [2, 3], decimal=1
        )


class TestEdgeCases:
    """边界情况测试类"""

    def test_empty_data(self):
        """测试空数据"""
        X = np.array([]).reshape(0, 2)
        with pytest.raises((IndexError, ValueError)):
            featureNormaliza(X)

    def test_single_sample(self):
        """测试单样本数据"""
        X = np.array([[1, 2]], dtype=np.float64)
        X_norm, mu, sigma = featureNormaliza(X)
        # 单样本归一化后应为0（因为等于均值）
        np.testing.assert_array_almost_equal(X_norm[0], [0, 0])

    def test_large_values(self):
        """测试大数值数据"""
        X = np.array([[1000, 2000], [2000, 4000], [3000, 6000]], dtype=np.float64)
        X_norm, mu, sigma = featureNormaliza(X)
        # 归一化后应在合理范围内
        assert np.all(np.abs(X_norm) < 10)


class TestSklearnComparison:
    """与 scikit-learn 对比测试类"""

    def test_compare_with_sklearn(self, sample_regression_data):
        """与 sklearn 线性回归结果对比"""
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            pytest.skip("scikit-learn not installed")
        
        X, y, _ = sample_regression_data
        
        # 自定义实现
        m = len(y)
        X_norm_custom, mu, sigma = featureNormaliza(X)
        X_norm_custom = np.hstack((np.ones((m, 1)), X_norm_custom))
        theta = np.zeros((X_norm_custom.shape[1], 1))
        theta_custom, _ = gradientDescent(X_norm_custom, y.reshape(-1, 1), 
                                          theta, 0.01, 500)
        
        # sklearn 实现
        scaler = StandardScaler()
        X_sklearn = scaler.fit_transform(X)
        model = LinearRegression()
        model.fit(X_sklearn, y)
        
        # 比较预测结果（允许一定误差）
        predictions_custom = np.dot(X_norm_custom, theta_custom)
        predictions_sklearn = model.predict(X_sklearn)
        
        correlation = np.corrcoef(predictions_custom.flatten(), 
                                  predictions_sklearn)[0, 1]
        assert correlation > 0.95  # 相关性应很高


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
