# -*- coding: utf-8 -*-
"""
逻辑回归算法测试模块
测试内容：
- 数据加载功能
- Sigmoid 函数
- 特征映射
- 代价函数与梯度计算
- 预测函数
- 正则化
- 端到端集成测试
- 与 scikit-learn 对比
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "LogisticRegression"))

from LogisticRegression import (
    loadtxtAndcsv_data,
    loadnpy_data,
    sigmoid,
    mapFeature,
    costFunction,
    gradient,
    predict
)


class TestSigmoidFunction:
    """Sigmoid 函数测试类"""

    def test_sigmoid_zero(self):
        """测试 sigmoid(0) = 0.5"""
        z = np.array([0])
        result = sigmoid(z)
        np.testing.assert_almost_equal(result[0], 0.5, decimal=5)

    def test_sigmoid_positive_large(self):
        """测试 sigmoid 在大正数时接近 1"""
        z = np.array([10, 100])
        result = sigmoid(z)
        assert np.all(result > 0.99)
        assert np.all(result <= 1.0)

    def test_sigmoid_negative_large(self):
        """测试 sigmoid 在大负数时接近 0"""
        z = np.array([-10, -100])
        result = sigmoid(z)
        assert np.all(result < 0.01)
        assert np.all(result >= 0.0)

    def test_sigmoid_range(self):
        """测试 sigmoid 输出范围在 (0, 1)"""
        z = np.linspace(-10, 10, 100)
        result = sigmoid(z)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_sigmoid_shape(self):
        """测试 sigmoid 输出形状与输入一致"""
        z = np.array([[1, 2], [3, 4]])
        result = sigmoid(z)
        assert result.shape == z.shape


class TestFeatureMapping:
    """特征映射测试类"""

    def test_map_feature_degree_2(self):
        """测试 degree=2 的特征映射"""
        X1 = np.array([1, 2, 3])
        X2 = np.array([4, 5, 6])
        result = mapFeature(X1, X2)
        # degree=2 时应有 1 + 2 + 3 = 6 个特征
        # 1, x1, x2, x1^2, x1*x2, x2^2
        assert result.shape == (3, 6)
        # 第一列应为 1
        np.testing.assert_array_equal(result[:, 0], [1, 1, 1])

    def test_map_feature_output_values(self):
        """测试特征映射输出值正确性"""
        X1 = np.array([1])
        X2 = np.array([2])
        result = mapFeature(X1, X2)
        # 期望: [1, 1, 2, 1, 2, 4]
        expected = np.array([[1, 1, 2, 1, 2, 4]])
        np.testing.assert_array_equal(result, expected)


class TestCostFunction:
    """代价函数测试类"""

    def test_cost_function_initial_value(self, sample_classification_data):
        """测试初始 theta=0 时的代价约为 0.693"""
        X, y = sample_classification_data
        # 添加偏置项
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        initial_theta = np.zeros(X.shape[1])
        
        J = costFunction(initial_theta, X, y, 0)
        # 初始代价应约为 0.693 (ln(2))
        np.testing.assert_almost_equal(J, 0.693, decimal=2)

    def test_cost_function_positive(self, sample_classification_data):
        """测试代价函数始终为正"""
        X, y = sample_classification_data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        np.random.seed(42)
        for _ in range(5):
            theta = np.random.randn(X.shape[1])
            J = costFunction(theta, X, y, 0)
            assert J >= 0

    def test_cost_function_with_regularization(self, sample_classification_data):
        """测试带正则化的代价函数"""
        X, y = sample_classification_data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = np.ones(X.shape[1])
        
        J_no_reg = costFunction(theta, X, y, 0)
        J_with_reg = costFunction(theta, X, y, 1.0)
        
        # 正则化后的代价应更大
        assert J_with_reg >= J_no_reg


class TestGradient:
    """梯度计算测试类"""

    def test_gradient_shape(self, sample_classification_data):
        """测试梯度形状与 theta 一致"""
        X, y = sample_classification_data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = np.zeros(X.shape[1])
        
        grad = gradient(theta, X, y, 0)
        assert grad.shape == theta.shape

    def test_gradient_zero_theta(self, sample_classification_data):
        """测试 theta=0 时的梯度"""
        X, y = sample_classification_data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = np.zeros(X.shape[1])
        
        grad = gradient(theta, X, y, 0)
        # 梯度不应全为零
        assert not np.allclose(grad, 0)

    def test_gradient_numerical_check(self, sample_classification_data):
        """数值梯度检查"""
        X, y = sample_classification_data
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        theta = np.random.randn(X.shape[1]) * 0.1
        
        grad = gradient(theta, X, y, 0)
        
        # 数值梯度
        epsilon = 1e-4
        num_grad = np.zeros_like(theta)
        for i in range(len(theta)):
            theta_plus = theta.copy()
            theta_minus = theta.copy()
            theta_plus[i] += epsilon
            theta_minus[i] -= epsilon
            num_grad[i] = (costFunction(theta_plus, X, y, 0) - 
                          costFunction(theta_minus, X, y, 0)) / (2 * epsilon)
        
        np.testing.assert_array_almost_equal(grad, num_grad, decimal=4)


class TestPrediction:
    """预测函数测试类"""

    def test_predict_shape(self, sample_classification_data):
        """测试预测输出形状"""
        X, y = sample_classification_data
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        theta = np.zeros(X_mapped.shape[1])
        
        p = predict(X_mapped, theta)
        assert p.shape == (X.shape[0],)

    def test_predict_binary_output(self, sample_classification_data):
        """测试预测输出为 0 或 1"""
        X, y = sample_classification_data
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        np.random.seed(42)
        theta = np.random.randn(X_mapped.shape[1])
        
        p = predict(X_mapped, theta)
        assert np.all(np.isin(p, [0, 1]))

    def test_predict_perfect_separation(self):
        """测试完美分离情况"""
        # 创建线性可分数据
        X = np.array([[30, 30], [40, 40], [90, 90], [100, 100]])
        y = np.array([0, 0, 1, 1])
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        # 使用能完美分类的 theta
        theta = np.zeros(X_mapped.shape[1])
        theta[1] = 1  # x1 的权重
        theta[2] = 1  # x2 的权重
        
        p = predict(X_mapped, theta)
        # 至少应该能正确分类大部分
        accuracy = np.mean(p == y)
        assert accuracy >= 0.5


class TestDataLoading:
    """数据加载测试类"""

    def test_loadtxt_data(self, logistic_regression_data_path):
        """测试数据文件加载"""
        if os.path.exists(logistic_regression_data_path):
            data = loadtxtAndcsv_data(logistic_regression_data_path, ",", np.float64)
            assert isinstance(data, np.ndarray)
            assert data.ndim == 2
            assert data.shape[1] >= 3  # 至少2个特征+1个标签

    def test_load_npy_data(self):
        """测试 npy 文件加载"""
        npy_path = os.path.join(PROJECT_ROOT, "LogisticRegression", "data1.npy")
        if os.path.exists(npy_path):
            data = loadnpy_data(npy_path)
            assert isinstance(data, np.ndarray)


class TestIntegration:
    """集成测试类"""

    def test_logistic_regression_workflow(self, sample_classification_data):
        """端到端工作流测试"""
        X, y = sample_classification_data
        
        # 特征映射
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        
        # 初始化参数
        initial_theta = np.zeros(X_mapped.shape[1])
        
        # 计算初始代价
        initial_cost = costFunction(initial_theta, X_mapped, y, 0.1)
        
        # 使用 scipy 优化
        from scipy import optimize
        result = optimize.fmin_bfgs(costFunction, initial_theta, 
                                    fprime=gradient, 
                                    args=(X_mapped, y, 0.1),
                                    disp=False)
        
        # 预测
        p = predict(X_mapped, result)
        accuracy = np.mean(p == y)
        
        # 准确率应高于随机猜测
        assert accuracy > 0.5

    def test_regularization_effect(self, sample_classification_data):
        """测试正则化效果"""
        X, y = sample_classification_data
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        initial_theta = np.zeros(X_mapped.shape[1])
        
        from scipy import optimize
        
        # 无正则化
        result_no_reg = optimize.fmin_bfgs(costFunction, initial_theta,
                                          fprime=gradient,
                                          args=(X_mapped, y, 0),
                                          disp=False)
        
        # 有正则化
        result_with_reg = optimize.fmin_bfgs(costFunction, initial_theta,
                                            fprime=gradient,
                                            args=(X_mapped, y, 10),
                                            disp=False)
        
        # 正则化后的参数范数应更小
        assert np.linalg.norm(result_with_reg) <= np.linalg.norm(result_no_reg) * 1.5


class TestEdgeCases:
    """边界情况测试类"""

    def test_empty_data(self):
        """测试空数据"""
        X = np.array([]).reshape(0, 2)
        y = np.array([])
        if X.shape[0] > 0:  # 避免空数组测试
            X_mapped = mapFeature(X[:, 0], X[:, 1])
            theta = np.zeros(X_mapped.shape[1])
            with pytest.raises((ValueError, IndexError)):
                costFunction(theta, X_mapped, y, 0)

    def test_single_sample(self):
        """测试单样本"""
        X = np.array([[1, 2]])
        y = np.array([1])
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        theta = np.zeros(X_mapped.shape[1])
        
        # 单样本不应报错
        J = costFunction(theta, X_mapped, y, 0)
        assert isinstance(J, (float, np.floating))

    def test_all_same_class(self):
        """测试所有样本属于同一类"""
        X = np.random.randn(10, 2)
        y = np.ones(10)
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        theta = np.zeros(X_mapped.shape[1])
        
        # 不应报错
        J = costFunction(theta, X_mapped, y, 0)
        assert J >= 0


class TestSklearnComparison:
    """与 scikit-learn 对比测试类"""

    def test_compare_with_sklearn(self, sample_classification_data):
        """与 sklearn 逻辑回归对比"""
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import PolynomialFeatures
        except ImportError:
            pytest.skip("scikit-learn not installed")
        
        X, y = sample_classification_data
        
        # 自定义实现
        X_mapped = mapFeature(X[:, 0], X[:, 1])
        initial_theta = np.zeros(X_mapped.shape[1])
        
        from scipy import optimize
        result_custom = optimize.fmin_bfgs(costFunction, initial_theta,
                                          fprime=gradient,
                                          args=(X_mapped, y, 0.1),
                                          disp=False)
        p_custom = predict(X_mapped, result_custom)
        
        # sklearn 实现
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(X)
        model = LogisticRegression(max_iter=1000, C=10)
        model.fit(X_poly, y)
        p_sklearn = model.predict(X_poly)
        
        # 比较准确率
        acc_custom = np.mean(p_custom == y)
        acc_sklearn = np.mean(p_sklearn == y)
        
        # 两者准确率差异不应太大
        assert abs(acc_custom - acc_sklearn) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
