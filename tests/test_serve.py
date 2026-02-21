"""
Unit tests for the serving API
"""

import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


# Mock the model and labels before importing serve
@pytest.fixture(autouse=True)
def mock_startup():
    with patch('serve.model') as mock_model, \
         patch('serve.labels', ['setosa', 'versicolor', 'virginica']):
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        mock_model.n_estimators = 100
        mock_model.max_depth = 5
        mock_model.n_features_in_ = 4
        yield


class TestServingAPI:
    """Test cases for the ML serving API."""
    
    def test_health_check(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            response = client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
    
    def test_readiness_probe(self):
        """Test Kubernetes readiness probe."""
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            response = client.get("/ready")
            
            assert response.status_code == 200
    
    def test_liveness_probe(self):
        """Test Kubernetes liveness probe."""
        from fastapi.testclient import TestClient
        from serve import app
        
        client = TestClient(app)
        response = client.get("/live")
        
        assert response.status_code == 200
    
    def test_predict_valid_input(self):
        """Test prediction with valid input."""
        from fastapi.testclient import TestClient
        from serve import app
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        
        with patch('serve.model', mock_model), \
             patch('serve.labels', ['setosa', 'versicolor', 'virginica']):
            client = TestClient(app)
            response = client.post(
                "/predict",
                json={"features": [5.1, 3.5, 1.4, 0.2]}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
    
    def test_predict_invalid_input(self):
        """Test prediction with invalid input (wrong number of features)."""
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            response = client.post(
                "/predict",
                json={"features": [5.1, 3.5]}  # Only 2 features instead of 4
            )
            
            assert response.status_code == 422  # Validation error
    
    def test_batch_predict(self):
        """Test batch prediction."""
        from fastapi.testclient import TestClient
        from serve import app
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0, 2])
        mock_model.predict_proba.return_value = np.array([
            [0.9, 0.05, 0.05],
            [0.1, 0.1, 0.8]
        ])
        
        with patch('serve.model', mock_model), \
             patch('serve.labels', ['setosa', 'versicolor', 'virginica']):
            client = TestClient(app)
            response = client.post(
                "/batch_predict",
                json={
                    "instances": [
                        [5.1, 3.5, 1.4, 0.2],
                        [6.7, 3.0, 5.2, 2.3]
                    ]
                }
            )
            
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        from fastapi.testclient import TestClient
        from serve import app
        
        client = TestClient(app)
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "endpoints" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
