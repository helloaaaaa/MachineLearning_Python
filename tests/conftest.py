# -*- coding: utf-8 -*-
"""
Pytest 共享 fixtures 配置文件
为所有机器学习算法测试提供公共的数据加载和模型初始化功能
"""
import pytest
import numpy as np
import os
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def sample_regression_data():
    """生成小型回归数据集用于测试"""
    np.random.seed(42)
    m = 50
    X = np.random.randn(m, 2)
    true_theta = np.array([2.0, -1.5])
    y = X[:, 0] * true_theta[0] + X[:, 1] * true_theta[1] + 0.1 * np.random.randn(m)
    return X, y, true_theta


@pytest.fixture
def sample_classification_data():
    """生成小型二分类数据集用于测试"""
    np.random.seed(42)
    m = 100
    # 生成两类数据
    X_pos = np.random.randn(m // 2, 2) + np.array([1.0, 1.0])
    X_neg = np.random.randn(m // 2, 2) - np.array([1.0, 1.0])
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(m // 2), np.zeros(m // 2)])
    return X, y


@pytest.fixture
def sample_multiclass_data():
    """生成小型多分类数据集用于测试"""
    np.random.seed(42)
    m = 150
    n_classes = 3
    X_list = []
    y_list = []
    for i in range(n_classes):
        center = np.array([np.cos(2 * np.pi * i / n_classes), 
                          np.sin(2 * np.pi * i / n_classes)]) * 2
        X_list.append(np.random.randn(m // n_classes, 2) + center)
        y_list.append(np.ones(m // n_classes) * i)
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    return X, y.astype(int)


@pytest.fixture
def sample_clustering_data():
    """生成小型聚类数据集用于测试"""
    np.random.seed(42)
    K = 3
    m_per_cluster = 30
    centroids = np.array([[0, 0], [5, 5], [10, 0]])
    X_list = []
    for i in range(K):
        X_list.append(np.random.randn(m_per_cluster, 2) + centroids[i])
    X = np.vstack(X_list)
    return X, K, centroids


@pytest.fixture
def linear_regression_data_path():
    """线性回归数据文件路径"""
    return os.path.join(PROJECT_ROOT, "LinearRegression", "data.txt")


@pytest.fixture
def logistic_regression_data_path():
    """逻辑回归数据文件路径"""
    return os.path.join(PROJECT_ROOT, "LogisticRegression", "data2.txt")


@pytest.fixture
def kmeans_data_path():
    """K-Means 数据文件路径"""
    return os.path.join(PROJECT_ROOT, "K-Means", "data.mat")


@pytest.fixture
def pca_data_path():
    """PCA 数据文件路径"""
    return os.path.join(PROJECT_ROOT, "PCA", "data.mat")


@pytest.fixture
def anomaly_detection_data_path():
    """异常检测数据文件路径"""
    return os.path.join(PROJECT_ROOT, "AnomalyDetection", "data1.mat")


@pytest.fixture
def neural_network_data_path():
    """神经网络数据文件路径"""
    return os.path.join(PROJECT_ROOT, "NeuralNetwok", "data_digits.mat")


@pytest.fixture
def svm_data_path():
    """SVM 数据文件路径"""
    return os.path.join(PROJECT_ROOT, "SVM", "data1.mat")
