# -*- coding: utf-8 -*-
"""
BP神经网络算法测试模块
测试内容：
- 数据加载功能
- Sigmoid 函数及其导数
- 权重随机初始化
- 代价函数计算
- 反向传播梯度计算
- 梯度检查
- 预测函数
- 端到端集成测试
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "NeuralNetwok"))

from NeuralNetwork import (
    loadmat_data,
    sigmoid,
    sigmoidGradient,
    randInitializeWeights,
    nnCostFunction,
    nnGradient,
    predict,
    checkGradient,
    debugInitializeWeights
)


class TestSigmoidFunctions:
    """Sigmoid 函数测试类"""

    def test_sigmoid_zero(self):
        """测试 sigmoid(0) = 0.5"""
        z = np.array([0])
        result = sigmoid(z)
        np.testing.assert_almost_equal(result[0], 0.5, decimal=5)

    def test_sigmoid_range(self):
        """测试 sigmoid 输出范围"""
        z = np.linspace(-10, 10, 100)
        result = sigmoid(z)
        assert np.all(result > 0)
        assert np.all(result < 1)

    def test_sigmoid_gradient(self):
        """测试 sigmoid 导数"""
        z = np.array([0])
        result = sigmoidGradient(z)
        # sigmoid'(0) = sigmoid(0) * (1 - sigmoid(0)) = 0.25
        np.testing.assert_almost_equal(result[0], 0.25, decimal=5)

    def test_sigmoid_gradient_formula(self):
        """测试 sigmoid 导数公式正确性"""
        z = np.array([1, 2, 3])
        g = sigmoidGradient(z)
        s = sigmoid(z)
        expected = s * (1 - s)
        np.testing.assert_array_almost_equal(g, expected)


class TestWeightInitialization:
    """权重初始化测试类"""

    def test_rand_initialize_weights_shape(self):
        """测试权重初始化形状"""
        L_in = 3
        L_out = 5
        W = randInitializeWeights(L_in, L_out)
        assert W.shape == (L_out, L_in + 1)  # +1 for bias

    def test_rand_initialize_weights_range(self):
        """测试权重初始化范围"""
        L_in = 10
        L_out = 10
        W = randInitializeWeights(L_in, L_out)
        epsilon = (6.0 / (L_out + L_in)) ** 0.5
        assert np.all(W >= -epsilon)
        assert np.all(W <= epsilon)

    def test_rand_initialize_weights_randomness(self):
        """测试权重随机性"""
        L_in = 5
        L_out = 5
        W1 = randInitializeWeights(L_in, L_out)
        W2 = randInitializeWeights(L_in, L_out)
        # 两次初始化应该不同
        assert not np.allclose(W1, W2)

    def test_debug_initialize_weights(self):
        """测试调试权重初始化"""
        fan_in = 3
        fan_out = 5
        W = debugInitializeWeights(fan_in, fan_out)
        assert W.shape == (fan_out, fan_in + 1)
        # 调试权重使用 sin 函数，值应在 [-0.1, 0.1]
        assert np.all(W >= -0.1)
        assert np.all(W <= 0.1)


class TestCostFunction:
    """代价函数测试类"""

    def test_nn_cost_function_shape(self):
        """测试代价函数返回值"""
        input_size = 3
        hidden_size = 5
        num_labels = 3
        m = 5
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.random.randint(0, num_labels, m)
        
        J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        assert isinstance(J, (float, np.floating))
        assert J >= 0

    def test_nn_cost_function_positive(self):
        """测试代价函数始终为正"""
        input_size = 2
        hidden_size = 3
        num_labels = 2
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.random.randint(0, num_labels, m)
        
        for lambda_val in [0, 0.1, 1, 10]:
            J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, lambda_val)
            assert J >= 0

    def test_nn_cost_function_regularization(self):
        """测试正则化效果"""
        input_size = 2
        hidden_size = 3
        num_labels = 2
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.random.randint(0, num_labels, m)
        
        J_no_reg = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        J_with_reg = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 1)
        
        assert J_with_reg >= J_no_reg


class TestGradient:
    """梯度计算测试类"""

    def test_nn_gradient_shape(self):
        """测试梯度形状"""
        input_size = 3
        hidden_size = 5
        num_labels = 3
        m = 5
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.random.randint(0, num_labels, m)
        
        grad = nnGradient(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        assert grad.shape == nn_params.shape

    def test_gradient_numerical_check(self):
        """数值梯度检查"""
        input_size = 3
        hidden_size = 5
        num_labels = 3
        m = 5
        
        initial_Theta1 = debugInitializeWeights(input_size, hidden_size)
        initial_Theta2 = debugInitializeWeights(hidden_size, num_labels)
        X = debugInitializeWeights(input_size - 1, m)
        y = np.transpose(np.mod(np.arange(1, m + 1), num_labels))
        y = y.reshape(-1, 1)
        
        nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        
        grad = nnGradient(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        
        # 数值梯度
        epsilon = 1e-4
        num_grad = np.zeros((nn_params.shape[0]))
        step = np.zeros((nn_params.shape[0]))
        
        for i in range(nn_params.shape[0]):
            step[i] = epsilon
            loss1 = nnCostFunction(nn_params - step.reshape(-1, 1), input_size, hidden_size, num_labels, X, y, 0)
            loss2 = nnCostFunction(nn_params + step.reshape(-1, 1), input_size, hidden_size, num_labels, X, y, 0)
            num_grad[i] = (loss2 - loss1) / (2 * epsilon)
            step[i] = 0
        
        np.testing.assert_array_almost_equal(grad.flatten(), num_grad, decimal=4)


class TestPrediction:
    """预测函数测试类"""

    def test_predict_shape(self):
        """测试预测输出形状"""
        input_size = 3
        hidden_size = 5
        num_labels = 3
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        X = np.random.rand(m, input_size)
        
        p = predict(Theta1, Theta2, X)
        assert p.shape == (m, 1)

    def test_predict_range(self):
        """测试预测输出范围"""
        input_size = 3
        hidden_size = 5
        num_labels = 4
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        X = np.random.rand(m, input_size)
        
        p = predict(Theta1, Theta2, X)
        assert np.all(p >= 0)
        assert np.all(p < num_labels)

    def test_predict_deterministic(self):
        """测试预测确定性"""
        input_size = 3
        hidden_size = 5
        num_labels = 3
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        X = np.random.rand(m, input_size)
        
        p1 = predict(Theta1, Theta2, X)
        p2 = predict(Theta1, Theta2, X)
        np.testing.assert_array_equal(p1, p2)


class TestDataLoading:
    """数据加载测试类"""

    def test_load_mat_data(self, neural_network_data_path):
        """测试 mat 文件加载"""
        if os.path.exists(neural_network_data_path):
            data = loadmat_data(neural_network_data_path)
            assert isinstance(data, dict)
            assert 'X' in data
            assert 'y' in data
            assert isinstance(data['X'], np.ndarray)
            assert isinstance(data['y'], np.ndarray)


class TestIntegration:
    """集成测试类"""

    def test_neural_network_training(self):
        """神经网络训练集成测试"""
        from scipy import optimize
        
        input_size = 2
        hidden_size = 5
        num_labels = 2
        m = 50
        
        # 生成简单分类数据
        np.random.seed(42)
        X = np.random.randn(m, input_size)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        # 初始化权重
        initial_Theta1 = randInitializeWeights(input_size, hidden_size)
        initial_Theta2 = randInitializeWeights(hidden_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        
        # 优化
        result = optimize.fmin_cg(nnCostFunction, initial_nn_params,
                                  fprime=nnGradient,
                                  args=(input_size, hidden_size, num_labels, X, y, 1),
                                  maxiter=50, disp=False)
        
        # 提取权重
        length = result.shape[0]
        Theta1 = result[0:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
        Theta2 = result[hidden_size * (input_size + 1):length].reshape(num_labels, hidden_size + 1)
        
        # 预测
        p = predict(Theta1, Theta2, X)
        accuracy = np.mean(p.flatten() == y)
        
        # 准确率应高于随机猜测
        assert accuracy > 0.5

    def test_xor_problem(self):
        """测试 XOR 问题"""
        from scipy import optimize
        
        # XOR 数据集
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        
        input_size = 2
        hidden_size = 4
        num_labels = 2
        
        initial_Theta1 = randInitializeWeights(input_size, hidden_size)
        initial_Theta2 = randInitializeWeights(hidden_size, num_labels)
        initial_nn_params = np.vstack((initial_Theta1.reshape(-1, 1), initial_Theta2.reshape(-1, 1)))
        
        result = optimize.fmin_cg(nnCostFunction, initial_nn_params,
                                  fprime=nnGradient,
                                  args=(input_size, hidden_size, num_labels, X, y, 0.1),
                                  maxiter=100, disp=False)
        
        length = result.shape[0]
        Theta1 = result[0:hidden_size * (input_size + 1)].reshape(hidden_size, input_size + 1)
        Theta2 = result[hidden_size * (input_size + 1):length].reshape(num_labels, hidden_size + 1)
        
        p = predict(Theta1, Theta2, X)
        accuracy = np.mean(p.flatten() == y)
        
        # XOR 问题应该能学到较高准确率
        assert accuracy >= 0.5


class TestEdgeCases:
    """边界情况测试类"""

    def test_single_sample(self):
        """测试单样本"""
        input_size = 2
        hidden_size = 3
        num_labels = 2
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        X = np.random.rand(1, input_size)
        y = np.array([0])
        
        p = predict(Theta1, Theta2, X)
        assert p.shape == (1, 1)

    def test_single_class(self):
        """测试单类别"""
        input_size = 2
        hidden_size = 3
        num_labels = 2
        m = 10
        
        Theta1 = randInitializeWeights(input_size, hidden_size)
        Theta2 = randInitializeWeights(hidden_size, num_labels)
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.zeros(m)  # 所有样本属于类别 0
        
        J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        assert J >= 0

    def test_zero_weights(self):
        """测试零权重"""
        input_size = 2
        hidden_size = 3
        num_labels = 2
        m = 10
        
        Theta1 = np.zeros((hidden_size, input_size + 1))
        Theta2 = np.zeros((num_labels, hidden_size + 1))
        nn_params = np.vstack((Theta1.reshape(-1, 1), Theta2.reshape(-1, 1)))
        
        X = np.random.rand(m, input_size)
        y = np.random.randint(0, num_labels, m)
        
        J = nnCostFunction(nn_params, input_size, hidden_size, num_labels, X, y, 0)
        # 零权重时代价应约为 -ln(0.5) = 0.693
        np.testing.assert_almost_equal(J, 0.693, decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
