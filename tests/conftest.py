# -*- coding: utf-8 -*-
"""
Shared fixtures for machine learning algorithm tests.
"""
import os
import sys
import numpy as np
from scipy import io as spio
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def project_root():
    return PROJECT_ROOT


@pytest.fixture
def linear_regression_data_path(project_root):
    return os.path.join(project_root, "LinearRegression", "data.txt")


@pytest.fixture
def logistic_regression_data1_path(project_root):
    return os.path.join(project_root, "LogisticRegression", "data1.txt")


@pytest.fixture
def logistic_regression_data2_path(project_root):
    return os.path.join(project_root, "LogisticRegression", "data2.txt")


@pytest.fixture
def logistic_regression_digits_path(project_root):
    return os.path.join(project_root, "LogisticRegression", "data_digits.mat")


@pytest.fixture
def neural_network_digits_path(project_root):
    return os.path.join(project_root, "NeuralNetwok", "data_digits.mat")


@pytest.fixture
def svm_data1_path(project_root):
    return os.path.join(project_root, "SVM", "data1.mat")


@pytest.fixture
def svm_data2_path(project_root):
    return os.path.join(project_root, "SVM", "data2.mat")


@pytest.fixture
def svm_data3_path(project_root):
    return os.path.join(project_root, "SVM", "data3.mat")


@pytest.fixture
def kmeans_data_path(project_root):
    return os.path.join(project_root, "K-Means", "data.mat")


@pytest.fixture
def pca_data_path(project_root):
    return os.path.join(project_root, "PCA", "data.mat")


@pytest.fixture
def pca_faces_data_path(project_root):
    return os.path.join(project_root, "PCA", "data_faces.mat")


@pytest.fixture
def anomaly_detection_data1_path(project_root):
    return os.path.join(project_root, "AnomalyDetection", "data1.mat")


@pytest.fixture
def anomaly_detection_data2_path(project_root):
    return os.path.join(project_root, "AnomalyDetection", "data2.mat")


@pytest.fixture
def load_txt_data():
    def _load(path, delimiter=",", dtype=np.float64):
        return np.loadtxt(path, delimiter=delimiter, dtype=dtype)
    return _load


@pytest.fixture
def load_mat_data():
    def _load(path):
        return spio.loadmat(path)
    return _load


@pytest.fixture
def simple_regression_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    true_theta = np.array([[3.0], [2.0], [1.0]])
    X_with_bias = np.hstack((np.ones((100, 1)), X))
    y = np.dot(X_with_bias, true_theta) + np.random.randn(100, 1) * 0.1
    return X, y, true_theta


@pytest.fixture
def simple_classification_data():
    np.random.seed(42)
    X = np.random.randn(100, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def simple_multiclass_data():
    np.random.seed(42)
    X = np.random.randn(150, 2)
    y = np.zeros(150)
    y[50:100] = 1
    y[100:] = 2
    X[:50] += np.array([2, 2])
    X[50:100] += np.array([-2, 2])
    X[100:] += np.array([0, -2])
    return X, y


@pytest.fixture
def simple_clustering_data():
    np.random.seed(42)
    cluster1 = np.random.randn(30, 2) + np.array([0, 0])
    cluster2 = np.random.randn(30, 2) + np.array([5, 5])
    cluster3 = np.random.randn(30, 2) + np.array([5, 0])
    X = np.vstack((cluster1, cluster2, cluster3))
    return X


@pytest.fixture
def simple_pca_data():
    np.random.seed(42)
    X = np.random.randn(100, 5)
    return X


@pytest.fixture
def simple_anomaly_data():
    np.random.seed(42)
    X_normal = np.random.randn(100, 2)
    X_anomaly = np.random.randn(10, 2) + np.array([5, 5])
    X = np.vstack((X_normal, X_anomaly))
    y = np.zeros(110)
    y[100:] = 1
    return X, y
