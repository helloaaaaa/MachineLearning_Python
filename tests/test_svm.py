# -*- coding: utf-8 -*-
"""
Unit tests for SVM module.
Tests cover: data loading, linear SVM, non-linear SVM with kernels,
and comparison with different parameters.
"""
import os
import sys
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'SVM'))

from conftest import (
    SVM_DIR, assert_array_almost_equal_custom, assert_shape
)


class TestDataLoading:
    """Tests for data loading functionality"""

    def test_load_data1_exists(self, svm_data1):
        """Test that SVM data1 file can be loaded"""
        assert svm_data1 is not None, "Data1 file should exist"

    def test_load_data2_exists(self, svm_data2):
        """Test that SVM data2 file can be loaded"""
        assert svm_data2 is not None, "Data2 file should exist"

    def test_load_data1_shape(self, svm_data1):
        """Test that loaded data1 has correct shape"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        X = svm_data1['X']
        y = svm_data1['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"
        assert X.shape[1] == 2, "X should have 2 features"

    def test_load_data2_shape(self, svm_data2):
        """Test that loaded data2 has correct shape"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        X = svm_data2['X']
        y = svm_data2['y']
        assert X.shape[0] == y.shape[0], "X and y should have same number of samples"

    def test_data1_values(self, svm_data1):
        """Test that data1 values are valid"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        X = svm_data1['X']
        y = svm_data1['y']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isnan(y)), "y should not contain NaN values"
        assert set(np.unique(y)).issubset({0, 1}), "y should be binary (0 or 1)"

    def test_data2_values(self, svm_data2):
        """Test that data2 values are valid"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        X = svm_data2['X']
        y = svm_data2['y']
        assert not np.any(np.isnan(X)), "X should not contain NaN values"
        assert not np.any(np.isnan(y)), "y should not contain NaN values"


class TestLinearSVM:
    """Tests for linear SVM"""

    def test_linear_svm_training(self, svm_data1):
        """Test linear SVM training"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        assert model.support_vectors_ is not None, "Model should have support vectors"
        assert len(model.classes_) == 2, "Should be binary classification"

    def test_linear_svm_prediction_shape(self, svm_data1):
        """Test that prediction has correct shape"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == y.shape, "Predictions should have same shape as y"

    def test_linear_svm_accuracy(self, svm_data1):
        """Test linear SVM accuracy on training data"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.8, "Accuracy should be reasonable for linearly separable data"

    def test_linear_svm_coefficients(self, svm_data1):
        """Test that linear SVM produces coefficients"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        assert model.coef_ is not None, "Linear SVM should have coefficients"
        assert model.coef_.shape[1] == X.shape[1], "Coefficients should match feature count"

    def test_linear_svm_intercept(self, svm_data1):
        """Test that linear SVM produces intercept"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        assert model.intercept_ is not None, "Linear SVM should have intercept"


class TestNonLinearSVM:
    """Tests for non-linear SVM with kernels"""

    def test_rbf_svm_training(self, svm_data2):
        """Test RBF kernel SVM training"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        from sklearn import svm
        X = svm_data2['X']
        y = svm_data2['y']
        model = svm.SVC(C=1.0, kernel='rbf')
        model.fit(X, y)
        assert model.support_vectors_ is not None, "Model should have support vectors"

    def test_rbf_svm_accuracy(self, svm_data2):
        """Test RBF kernel SVM accuracy"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        from sklearn import svm
        X = svm_data2['X']
        y = svm_data2['y']
        model = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.7, "Accuracy should be reasonable"

    def test_rbf_gamma_effect(self, svm_data2):
        """Test effect of gamma parameter on RBF kernel"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        from sklearn import svm
        X = svm_data2['X']
        y = svm_data2['y']
        model_low_gamma = svm.SVC(C=1.0, kernel='rbf', gamma=1)
        model_low_gamma.fit(X, y)
        model_high_gamma = svm.SVC(C=1.0, kernel='rbf', gamma=100)
        model_high_gamma.fit(X, y)
        assert len(model_high_gamma.support_vectors_) <= len(model_low_gamma.support_vectors_) + 50, \
            "Higher gamma typically uses fewer or similar support vectors"

    def test_polynomial_kernel(self, svm_data2):
        """Test polynomial kernel SVM"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        from sklearn import svm
        X = svm_data2['X']
        y = svm_data2['y']
        model = svm.SVC(C=1.0, kernel='poly', degree=2)
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.5, "Polynomial kernel should achieve reasonable accuracy"


class TestSVMParameters:
    """Tests for SVM parameter effects"""

    def test_c_parameter_effect(self, svm_data1):
        """Test effect of C parameter"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model_low_c = svm.SVC(C=0.1, kernel='linear')
        model_low_c.fit(X, y)
        model_high_c = svm.SVC(C=100, kernel='linear')
        model_high_c.fit(X, y)
        acc_low = model_low_c.score(X, y)
        acc_high = model_high_c.score(X, y)
        assert acc_low > 0, "Low C should still achieve some accuracy"
        assert acc_high > 0, "High C should achieve some accuracy"

    def test_decision_function_shape(self, svm_data1):
        """Test decision function output shape"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        decision = model.decision_function(X)
        assert decision.shape[0] == X.shape[0], "Decision function should have same number of samples"

    def test_support_vectors_count(self, svm_data1):
        """Test that support vectors are reasonable fraction of data"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        n_support = len(model.support_vectors_)
        n_samples = X.shape[0]
        assert n_support < n_samples, "Support vectors should be subset of training data"
        assert n_support > 0, "Should have at least some support vectors"


class TestSVMPrediction:
    """Tests for SVM prediction functionality"""

    def test_predict_binary_output(self, svm_data1):
        """Test that prediction produces binary output"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        predictions = model.predict(X)
        assert set(np.unique(predictions)).issubset({0, 1}), "Predictions should be binary"

    def test_predict_proba_not_available_by_default(self, svm_data1):
        """Test that predict_proba requires probability=True"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        assert not hasattr(model, 'predict_proba') or model.probability == False, \
            "predict_proba should not be available without probability=True"

    def test_predict_proba_with_probability(self, svm_data1):
        """Test that predict_proba works with probability=True"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear', probability=True)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (X.shape[0], 2), "Probabilities should have shape (n_samples, 2)"
        assert np.allclose(proba.sum(axis=1), 1), "Probabilities should sum to 1"


class TestSVMIntegration:
    """Integration tests for SVM"""

    def test_svm_end_to_end_linear(self, svm_data1):
        """Test complete SVM pipeline with linear kernel"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        X = svm_data1['X']
        y = svm_data1['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        assert accuracy > 0.5, "Test accuracy should be reasonable"

    def test_svm_end_to_end_rbf(self, svm_data2):
        """Test complete SVM pipeline with RBF kernel"""
        if svm_data2 is None:
            pytest.skip("Data2 file not available")
        from sklearn import svm
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        X = svm_data2['X']
        y = svm_data2['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
        model.fit(X_train_scaled, y_train)
        accuracy = model.score(X_test_scaled, y_test)
        assert accuracy > 0.5, "Test accuracy should be reasonable"


class TestEdgeCases:
    """Tests for edge cases and boundary conditions"""

    def test_single_sample_prediction(self, svm_data1):
        """Test prediction on single sample"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        single_prediction = model.predict(X[0:1])
        assert single_prediction.shape == (1,), "Single prediction should have shape (1,)"

    def test_imbalanced_classes(self):
        """Test SVM with imbalanced classes"""
        from sklearn import svm
        np.random.seed(42)
        X_class0 = np.random.randn(100, 2)
        X_class1 = np.random.randn(10, 2) + 3
        X = np.vstack((X_class0, X_class1))
        y = np.hstack((np.zeros(100), np.ones(10)))
        model = svm.SVC(C=1.0, kernel='linear', class_weight='balanced')
        model.fit(X, y)
        predictions = model.predict(X)
        assert len(np.unique(predictions)) > 0, "Should make predictions"

    def test_very_small_c(self, svm_data1):
        """Test SVM with very small C (strong regularization)"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=0.001, kernel='linear')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy >= 0, "Accuracy should be non-negative"

    def test_very_large_c(self, svm_data1):
        """Test SVM with very large C (weak regularization)"""
        if svm_data1 is None:
            pytest.skip("Data1 file not available")
        from sklearn import svm
        X = svm_data1['X']
        y = svm_data1['y']
        model = svm.SVC(C=1000, kernel='linear')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy >= 0, "Accuracy should be non-negative"


class TestSyntheticData:
    """Tests using synthetic data"""

    def test_linearly_separable_data(self):
        """Test SVM on clearly linearly separable data"""
        from sklearn import svm
        np.random.seed(42)
        X = np.vstack((
            np.random.randn(50, 2) - 2,
            np.random.randn(50, 2) + 2
        ))
        y = np.hstack((np.zeros(50), np.ones(50)))
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.95, "Should achieve high accuracy on linearly separable data"

    def test_circular_data(self):
        """Test SVM on circular data with RBF kernel"""
        from sklearn import svm
        np.random.seed(42)
        n = 100
        theta = np.random.rand(n) * 2 * np.pi
        r_inner = 1
        r_outer = 3
        X_inner = np.column_stack((
            r_inner * np.cos(theta[:50]) + np.random.randn(50) * 0.1,
            r_inner * np.sin(theta[:50]) + np.random.randn(50) * 0.1
        ))
        X_outer = np.column_stack((
            r_outer * np.cos(theta[50:]) + np.random.randn(50) * 0.1,
            r_outer * np.sin(theta[50:]) + np.random.randn(50) * 0.1
        ))
        X = np.vstack((X_inner, X_outer))
        y = np.hstack((np.zeros(50), np.ones(50)))
        model = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
        model.fit(X, y)
        accuracy = model.score(X, y)
        assert accuracy > 0.9, "RBF kernel should handle circular data well"
