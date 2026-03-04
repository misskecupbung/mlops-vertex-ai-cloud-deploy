"""Tests for serving API."""

import json
import pytest
from unittest.mock import patch, MagicMock
import numpy as np


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
    """API endpoint tests."""
    
    def test_health_check(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            resp = client.get("/health")
            
            assert resp.status_code == 200
            assert resp.json()["status"] == "healthy"
            assert resp.json()["model_loaded"] is True
    
    def test_readiness_probe(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            assert client.get("/ready").status_code == 200
    
    def test_liveness_probe(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        client = TestClient(app)
        assert client.get("/live").status_code == 200
    
    def test_predict_valid_input(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.05, 0.05]])
        
        with patch('serve.model', mock_model), \
             patch('serve.labels', ['setosa', 'versicolor', 'virginica']):
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [5.1, 3.5, 1.4, 0.2]})
            
            assert resp.status_code == 200
            data = resp.json()
            assert "prediction" in data
            assert "confidence" in data
            assert "probabilities" in data
    
    def test_predict_invalid_input(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        with patch('serve.model', MagicMock()):
            client = TestClient(app)
            # only 2 features instead of 4
            resp = client.post("/predict", json={"features": [5.1, 3.5]})
            assert resp.status_code == 422
    
    def test_batch_predict(self):
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
            resp = client.post("/batch_predict", json={
                "instances": [[5.1, 3.5, 1.4, 0.2], [6.7, 3.0, 5.2, 2.3]]
            })
            
            assert resp.status_code == 200
            assert len(resp.json()["predictions"]) == 2
    
    def test_root_endpoint(self):
        from fastapi.testclient import TestClient
        from serve import app
        
        client = TestClient(app)
        resp = client.get("/")
        
        assert resp.status_code == 200
        data = resp.json()
        assert "service" in data
        assert "endpoints" in data


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
