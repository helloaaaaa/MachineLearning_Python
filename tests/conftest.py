# -*- coding: utf-8 -*-
"""
Shared fixtures for machine learning algorithm tests.
This module provides common test data and utilities across all test modules.
"""
import os
import sys
import numpy as np
import pytest
import scipy.io as spio

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LINEAR_REGRESSION_DIR = os.path.join(PROJECT_ROOT, 'LinearRegression')
LOGISTIC_REGRESSION_DIR = os.path.join(PROJECT_ROOT, 'LogisticRegression')
NEURAL_NETWORK_DIR = os.path.join(PROJECT_ROOT, 'NeuralNetwok')
SVM_DIR = os.path.join(PROJECT_ROOT, 'SVM')
KMEANS_DIR = os.path.join(PROJECT_ROOT, 'K-Means')
PCA_DIR = os.path.join(PROJECT_ROOT, 'PCA')
ANOMALY_DETECTION_DIR = os.path.join(PROJECT_ROOT, 'AnomalyDetection')


@pytest.fixture
def linear_regression_data():
    """Load linear regression data from data.txt"""
    data_path = os.path.join(LINEAR_REGRESSION_DIR, 'data.txt')
    if os.path.exists(data_path):
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        return {'X': X, 'y': y, 'data': data}
    return None


@pytest.fixture
def logistic_regression_data1():
    """Load logistic regression data1"""
    data_path = os.path.join(LOGISTIC_REGRESSION_DIR, 'data1.txt')
    if os.path.exists(data_path):
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        return {'X': X, 'y': y, 'data': data}
    return None


@pytest.fixture
def logistic_regression_data2():
    """Load logistic regression data2 for polynomial features"""
    data_path = os.path.join(LOGISTIC_REGRESSION_DIR, 'data2.txt')
    if os.path.exists(data_path):
        data = np.loadtxt(data_path, delimiter=',', dtype=np.float64)
        X = data[:, 0:-1]
        y = data[:, -1]
        return {'X': X, 'y': y, 'data': data}
    return None


@pytest.fixture
def digits_data():
    """Load handwritten digits data for OneVsAll and Neural Network"""
    data_path = os.path.join(LOGISTIC_REGRESSION_DIR, 'data_digits.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X'], 'y': data['y']}
    return None


@pytest.fixture
def svm_data1():
    """Load SVM data1 for linear classification"""
    data_path = os.path.join(SVM_DIR, 'data1.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X'], 'y': np.ravel(data['y'])}
    return None


@pytest.fixture
def svm_data2():
    """Load SVM data2 for non-linear classification"""
    data_path = os.path.join(SVM_DIR, 'data2.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X'], 'y': np.ravel(data['y'])}
    return None


@pytest.fixture
def kmeans_data():
    """Load K-Means clustering data"""
    data_path = os.path.join(KMEANS_DIR, 'data.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X']}
    return None


@pytest.fixture
def pca_data():
    """Load PCA 2D data"""
    data_path = os.path.join(PCA_DIR, 'data.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X']}
    return None


@pytest.fixture
def pca_face_data():
    """Load PCA face image data"""
    data_path = os.path.join(PCA_DIR, 'data_faces.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {'X': data['X']}
    return None


@pytest.fixture
def anomaly_detection_data():
    """Load anomaly detection data"""
    data_path = os.path.join(ANOMALY_DETECTION_DIR, 'data1.mat')
    if os.path.exists(data_path):
        data = spio.loadmat(data_path)
        return {
            'X': data['X'],
            'Xval': data['Xval'],
            'yval': data['yval']
        }
    return None


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    X = np.random.randn(n_samples, n_features)
    true_theta = np.array([1.5, -2.0, 1.0])
    y = np.dot(X, true_theta) + np.random.randn(n_samples) * 0.1
    return {'X': X, 'y': y, 'true_theta': true_theta}


@pytest.fixture
def synthetic_classification_data():
    """Generate synthetic classification data for testing"""
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float64)
    return {'X': X, 'y': y}


@pytest.fixture
def synthetic_clustering_data():
    """Generate synthetic clustering data for testing"""
    np.random.seed(42)
    n_samples = 150
    centers = np.array([[0, 0], [5, 5], [0, 5]])
    X = np.vstack([
        np.random.randn(50, 2) + centers[0],
        np.random.randn(50, 2) + centers[1],
        np.random.randn(50, 2) + centers[2]
    ])
    return {'X': X, 'n_clusters': 3}


@pytest.fixture
def small_neural_network_params():
    """Parameters for small neural network testing"""
    return {
        'input_layer_size': 3,
        'hidden_layer_size': 5,
        'num_labels': 3,
        'm': 5
    }


def assert_array_almost_equal_custom(actual, desired, decimal=6):
    """Custom assertion for numpy arrays with better error messages"""
    np.testing.assert_array_almost_equal(
        actual, desired, decimal=decimal,
        err_msg=f"Arrays are not almost equal up to {decimal} decimals"
    )


def assert_shape(actual, expected_shape):
    """Assert that array has expected shape"""
    assert actual.shape == expected_shape, \
        f"Expected shape {expected_shape}, got {actual.shape}"
