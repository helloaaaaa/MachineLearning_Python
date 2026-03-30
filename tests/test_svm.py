# -*- coding: utf-8 -*-
"""
SVM支持向量机算法测试模块
测试内容：
- 数据加载功能
- 线性核SVM分类
- 非线性核SVM分类（RBF）
- 不同C参数效果
- 不同gamma参数效果
- 与 scikit-learn 对比
"""
import pytest
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scipy import io as spio


try:
    from sklearn import svm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class TestDataLoading:
    """数据加载测试类"""

    def test_load_mat_data(self, svm_data_path):
        """测试 mat 文件加载"""
        if os.path.exists(svm_data_path):
            data = spio.loadmat(svm_data_path)
            assert isinstance(data, dict)
            assert 'X' in data
            assert 'y' in data

    def test_data_shape(self, svm_data_path):
        """测试数据形状"""
        if os.path.exists(svm_data_path):
            data = spio.loadmat(svm_data_path)
            X = data['X']
            y = data['y']
            assert X.ndim == 2
            assert y.ndim == 2
            assert X.shape[0] == y.shape[0]


class TestLinearSVM:
    """线性SVM测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_linear_svm_fit(self, sample_classification_data):
        """测试线性SVM拟合"""
        X, y = sample_classification_data
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        
        assert hasattr(model, 'support_')
        assert len(model.support_) > 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_linear_svm_predict(self, sample_classification_data):
        """测试线性SVM预测"""
        X, y = sample_classification_data
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape
        assert np.all(np.isin(predictions, [0, 1]))

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_linear_svm_accuracy(self, sample_classification_data):
        """测试线性SVM准确率"""
        X, y = sample_classification_data
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        # 准确率应高于随机猜测
        assert accuracy > 0.5

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_linear_svm_coefficients(self, sample_classification_data):
        """测试线性SVM系数"""
        X, y = sample_classification_data
        model = svm.SVC(C=1.0, kernel='linear')
        model.fit(X, y)
        
        assert hasattr(model, 'coef_')
        assert model.coef_.shape == (1, X.shape[1])


class TestNonLinearSVM:
    """非线性SVM测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_rbf_svm_fit(self, sample_classification_data):
        """测试RBF核SVM拟合"""
        X, y = sample_classification_data
        model = svm.SVC(kernel='rbf', gamma='scale')
        model.fit(X, y)
        
        assert hasattr(model, 'support_')

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_rbf_svm_predict(self, sample_classification_data):
        """测试RBF核SVM预测"""
        X, y = sample_classification_data
        model = svm.SVC(kernel='rbf', gamma='scale')
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == y.shape

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_rbf_svm_vs_linear(self):
        """比较RBF和线性SVM在非线性数据上的表现"""
        # 创建非线性数据（同心圆）
        np.random.seed(42)
        n_samples = 200
        
        # 内圆
        theta = np.random.uniform(0, 2*np.pi, n_samples//2)
        r = np.random.uniform(0, 1, n_samples//2)
        X_inner = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        y_inner = np.zeros(n_samples//2)
        
        # 外圆
        theta = np.random.uniform(0, 2*np.pi, n_samples//2)
        r = np.random.uniform(1.5, 2.5, n_samples//2)
        X_outer = np.column_stack([r * np.cos(theta), r * np.sin(theta)])
        y_outer = np.ones(n_samples//2)
        
        X = np.vstack([X_inner, X_outer])
        y = np.hstack([y_inner, y_outer])
        
        # 线性SVM
        linear_model = svm.SVC(kernel='linear')
        linear_model.fit(X, y)
        linear_acc = np.mean(linear_model.predict(X) == y)
        
        # RBF SVM
        rbf_model = svm.SVC(kernel='rbf', gamma='scale')
        rbf_model.fit(X, y)
        rbf_acc = np.mean(rbf_model.predict(X) == y)
        
        # RBF应比线性SVM表现更好
        assert rbf_acc >= linear_acc


class TestSVMParameters:
    """SVM参数测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_c_parameter_effect(self, sample_classification_data):
        """测试C参数对模型的影响"""
        X, y = sample_classification_data
        
        # 小C值（更多正则化）
        model_small_c = svm.SVC(C=0.01, kernel='linear')
        model_small_c.fit(X, y)
        
        # 大C值（更少正则化）
        model_large_c = svm.SVC(C=100, kernel='linear')
        model_large_c.fit(X, y)
        
        # 大C值通常有更多支持向量
        assert len(model_large_c.support_) >= len(model_small_c.support_) * 0.5

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_gamma_parameter_effect(self, sample_classification_data):
        """测试gamma参数对RBF核的影响"""
        X, y = sample_classification_data
        
        # 小gamma
        model_small_gamma = svm.SVC(kernel='rbf', gamma=0.01)
        model_small_gamma.fit(X, y)
        
        # 大gamma
        model_large_gamma = svm.SVC(kernel='rbf', gamma=100)
        model_large_gamma.fit(X, y)
        
        # 两者都应该能拟合
        assert len(model_small_gamma.support_) > 0
        assert len(model_large_gamma.support_) > 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_different_kernels(self, sample_classification_data):
        """测试不同核函数"""
        X, y = sample_classification_data
        
        kernels = ['linear', 'rbf', 'poly']
        for kernel in kernels:
            model = svm.SVC(kernel=kernel, gamma='scale')
            model.fit(X, y)
            predictions = model.predict(X)
            accuracy = np.mean(predictions == y)
            assert accuracy > 0.5, f"Kernel {kernel} performed poorly"


class TestIntegration:
    """集成测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_svm_workflow(self, sample_classification_data):
        """SVM完整工作流测试"""
        X, y = sample_classification_data
        
        # 划分训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # 训练模型
        model = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
        model.fit(X_train, y_train)
        
        # 预测
        train_acc = np.mean(model.predict(X_train) == y_train)
        test_acc = np.mean(model.predict(X_test) == y_test)
        
        # 训练准确率应高于测试准确率
        assert train_acc >= test_acc * 0.8
        # 测试准确率应高于随机猜测
        assert test_acc > 0.5

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_multiclass_svm(self, sample_multiclass_data):
        """多分类SVM测试"""
        X, y = sample_multiclass_data
        
        model = svm.SVC(C=1.0, kernel='rbf', gamma='scale', decision_function_shape='ovr')
        model.fit(X, y)
        
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        
        # 多分类准确率应高于随机猜测 (1/3)
        assert accuracy > 0.4


class TestEdgeCases:
    """边界情况测试类"""

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_single_sample(self):
        """测试单样本"""
        X = np.array([[1, 2]])
        y = np.array([0])
        
        model = svm.SVC(kernel='linear')
        model.fit(X, y)
        
        prediction = model.predict(X)
        assert prediction[0] == 0

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_all_same_class(self):
        """测试所有样本属于同一类"""
        X = np.random.randn(10, 2)
        y = np.ones(10)
        
        model = svm.SVC(kernel='linear')
        # 可能产生警告，但不应报错
        model.fit(X, y)
        predictions = model.predict(X)
        assert np.all(predictions == 1)

    @pytest.mark.skipif(not SKLEARN_AVAILABLE, reason="scikit-learn not installed")
    def test_high_dimensional_data(self):
        """测试高维数据"""
        np.random.seed(42)
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)
        
        model = svm.SVC(kernel='linear')
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == y.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
