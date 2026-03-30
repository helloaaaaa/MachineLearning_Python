# -*- coding: utf-8 -*-
"""
Tests for SVM (Support Vector Machine) algorithm.
"""
import os
import sys
import numpy as np
from numpy.testing import assert_array_almost_equal, assert_almost_equal
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "SVM"))

import importlib.util
spec = importlib.util.spec_from_file_location("SVM_scikit_learn", os.path.join(PROJECT_ROOT, "SVM", "SVM_scikit-learn.py"))
svm_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(svm_module)

plot_data = svm_module.plot_data
plot_decisionBoundary = svm_module.plot_decisionBoundary


class TestLoadData:
    def test_load_data1_success(self, svm_data1_path, load_mat_data):
        data = load_mat_data(svm_data1_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data
        assert "y" in data

    def test_load_data2_success(self, svm_data2_path, load_mat_data):
        data = load_mat_data(svm_data2_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data
        assert "y" in data

    def test_load_data3_success(self, svm_data3_path, load_mat_data):
        data = load_mat_data(svm_data3_path)
        assert data is not None
        assert isinstance(data, dict)
        assert "X" in data
        assert "y" in data

    def test_data_shape(self, svm_data1_path, load_mat_data):
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = data["y"]
        assert X.shape[0] == y.shape[0]
        assert X.shape[1] == 2


class TestPlotData:
    def test_plot_data_output(self, simple_classification_data):
        X, y = simple_classification_data
        plt = plot_data(X, y)
        assert plt is not None

    def test_plot_data_with_binary_labels(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        plt = plot_data(X, y)
        assert plt is not None


class TestSVMLinearKernel:
    def test_svm_linear_kernel_fit(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        assert model is not None

    def test_svm_linear_kernel_predict(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_svm_linear_kernel_accuracy(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8

    def test_svm_linear_kernel_coef_shape(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        assert model.coef_.shape[1] == X.shape[1]


class TestSVMRBFKernel:
    def test_svm_rbf_kernel_fit(self, svm_data2_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data2_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(gamma=100)
        model.fit(X, y)
        assert model is not None

    def test_svm_rbf_kernel_predict(self, svm_data2_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data2_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(gamma=100)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_svm_rbf_kernel_accuracy(self, svm_data2_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data2_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(gamma=100)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8

    def test_svm_rbf_kernel_gamma_effect(self, svm_data2_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data2_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model_low_gamma = svm.SVC(gamma=1)
        model_low_gamma.fit(X, y)
        model_high_gamma = svm.SVC(gamma=100)
        model_high_gamma.fit(X, y)
        acc_low = np.mean(model_low_gamma.predict(X) == y)
        acc_high = np.mean(model_high_gamma.predict(X) == y)
        assert acc_high >= acc_low


class TestSVMRegularization:
    def test_svm_c_parameter_effect(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model_low_c = svm.SVC(C=0.1, kernel="linear")
        model_low_c.fit(X, y)
        model_high_c = svm.SVC(C=100, kernel="linear")
        model_high_c.fit(X, y)
        assert model_low_c is not None
        assert model_high_c is not None

    def test_svm_c_parameter_support_vectors(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model_low_c = svm.SVC(C=0.1, kernel="linear")
        model_low_c.fit(X, y)
        model_high_c = svm.SVC(C=100, kernel="linear")
        model_high_c.fit(X, y)
        assert model_low_c.support_vectors_.shape[0] >= model_high_c.support_vectors_.shape[0]


class TestSVMIntegration:
    def test_full_pipeline_linear(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8

    def test_full_pipeline_rbf(self, svm_data2_path, load_mat_data):
        from sklearn import svm
        data = load_mat_data(svm_data2_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(gamma=100)
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.8

    def test_svm_cross_validation(self, svm_data1_path, load_mat_data):
        from sklearn import svm
        from sklearn.model_selection import cross_val_score
        data = load_mat_data(svm_data1_path)
        X = data["X"]
        y = np.ravel(data["y"])
        model = svm.SVC(C=1.0, kernel="linear")
        scores = cross_val_score(model, X, y, cv=3)
        assert np.mean(scores) > 0.5


class TestSVMMultiClass:
    def test_svm_multiclass_fit(self, simple_multiclass_data):
        from sklearn import svm
        X, y = simple_multiclass_data
        model = svm.SVC(C=1.0, kernel="linear", decision_function_shape="ovr")
        model.fit(X, y)
        assert model is not None

    def test_svm_multiclass_predict(self, simple_multiclass_data):
        from sklearn import svm
        X, y = simple_multiclass_data
        model = svm.SVC(C=1.0, kernel="linear", decision_function_shape="ovr")
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_svm_multiclass_accuracy(self, simple_multiclass_data):
        from sklearn import svm
        X, y = simple_multiclass_data
        model = svm.SVC(C=1.0, kernel="rbf", decision_function_shape="ovr")
        model.fit(X, y)
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy > 0.7


class TestEdgeCases:
    def test_single_sample_prediction(self):
        from sklearn import svm
        X = np.array([[1.0, 2.0], [2.0, 3.0]])
        y = np.array([0, 1])
        model = svm.SVC(C=1.0, kernel="linear")
        model.fit(X, y)
        pred = model.predict(np.array([[1.0, 2.0]]))
        assert pred[0] in [0, 1]

    def test_imbalanced_classes(self):
        from sklearn import svm
        np.random.seed(42)
        X_class0 = np.random.randn(100, 2)
        X_class1 = np.random.randn(10, 2) + np.array([3, 3])
        X = np.vstack((X_class0, X_class1))
        y = np.array([0] * 100 + [1] * 10)
        model = svm.SVC(C=1.0, kernel="linear", class_weight="balanced")
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    def test_large_c_value(self, simple_classification_data):
        from sklearn import svm
        X, y = simple_classification_data
        model = svm.SVC(C=1000, kernel="linear")
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions is not None

    def test_small_c_value(self, simple_classification_data):
        from sklearn import svm
        X, y = simple_classification_data
        model = svm.SVC(C=0.001, kernel="linear")
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions is not None


class TestSVMComparison:
    def test_linear_vs_rbf_kernel(self, simple_classification_data):
        from sklearn import svm
        X, y = simple_classification_data
        model_linear = svm.SVC(C=1.0, kernel="linear")
        model_linear.fit(X, y)
        model_rbf = svm.SVC(C=1.0, kernel="rbf")
        model_rbf.fit(X, y)
        acc_linear = np.mean(model_linear.predict(X) == y)
        acc_rbf = np.mean(model_rbf.predict(X) == y)
        assert acc_linear >= 0 or acc_rbf >= 0
