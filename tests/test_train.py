"""
Unit tests for the training script
"""

import json
import os
import pickle
import tempfile
import pytest
import numpy as np
from sklearn.datasets import load_iris


class TestTrainingFunctions:
    """Test cases for ML training functions."""
    
    def test_load_data(self):
        """Test data loading."""
        from train import load_data
        
        X, y, feature_names, target_names = load_data()
        
        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert len(feature_names) == 4
        assert len(target_names) == 3
    
    def test_train_model(self):
        """Test model training."""
        from train import train_model
        
        iris = load_iris()
        X, y = iris.data[:100], iris.target[:100]
        
        model = train_model(X, y, n_estimators=10, random_state=42)
        
        assert model is not None
        assert hasattr(model, 'predict')
        assert hasattr(model, 'predict_proba')
    
    def test_evaluate_model(self):
        """Test model evaluation."""
        from train import train_model, evaluate_model
        
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Split manually
        X_train, X_test = X[:120], X[120:]
        y_train, y_test = y[:120], y[120:]
        
        model = train_model(X_train, y_train, n_estimators=10, random_state=42)
        metrics = evaluate_model(
            model, X_train, X_test, y_train, y_test, iris.target_names
        )
        
        assert 'train_accuracy' in metrics
        assert 'test_accuracy' in metrics
        assert 'test_f1_score' in metrics
        assert metrics['test_accuracy'] > 0.8
    
    def test_save_model_locally(self):
        """Test local model saving."""
        from train import train_model, save_model_locally
        
        iris = load_iris()
        model = train_model(iris.data, iris.target, n_estimators=10)
        metrics = {'accuracy': 0.95}
        
        with tempfile.TemporaryDirectory() as tmpdir:
            model_path, metrics_path, labels_path = save_model_locally(
                model, metrics, iris.target_names, tmpdir
            )
            
            # Check files exist
            assert os.path.exists(model_path)
            assert os.path.exists(metrics_path)
            assert os.path.exists(labels_path)
            
            # Check model can be loaded
            with open(model_path, 'rb') as f:
                loaded_model = pickle.load(f)
            
            assert loaded_model is not None
            
            # Check metrics
            with open(metrics_path, 'r') as f:
                loaded_metrics = json.load(f)
            
            assert loaded_metrics['accuracy'] == 0.95


class TestModelPredictions:
    """Test model prediction quality."""
    
    def test_model_predictions(self):
        """Test that model makes reasonable predictions."""
        from train import train_model
        
        iris = load_iris()
        model = train_model(iris.data, iris.target, n_estimators=50)
        
        # Test known examples
        setosa_example = np.array([[5.1, 3.5, 1.4, 0.2]])
        virginica_example = np.array([[6.7, 3.0, 5.2, 2.3]])
        
        setosa_pred = model.predict(setosa_example)[0]
        virginica_pred = model.predict(virginica_example)[0]
        
        # Setosa should be class 0
        assert setosa_pred == 0
        # Virginica should be class 2
        assert virginica_pred == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
