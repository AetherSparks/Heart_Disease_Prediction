def test_index_get(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Patient Risk Predictor" in response.data

def test_models_page(client):
    response = client.get("/models")
    assert response.status_code == 200

def test_api_models_endpoint(client):
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.get_json()
    assert "models" in data
    assert "feature_names" in data

def test_api_metrics_endpoint(client):
    response = client.get("/api/metrics")
    assert response.status_code == 200

def test_api_feature_importance_endpoint(client):
    response = client.get("/api/feature_importance")
    assert response.status_code == 200

def test_api_predict_missing_model(client):
    payload = {
        "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 3,
        "model": "NonExistentModel"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400

def test_api_predict_invalid_age(client):
    payload = {
        "age": -5, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
        "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
        "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 3,
        "model": "Random Forest"
    }
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400

def test_api_predict_missing_fields(client):
    payload = {"model": "Random Forest"}
    response = client.post("/api/predict", json=payload)
    assert response.status_code == 400

def test_form_post_missing_model(client):
    response = client.post("/", data={"age": "63", "sex": "1"})
    assert response.status_code == 400

def test_404_handler(client):
    response = client.get("/nonexistent")
    assert response.status_code == 404
