from flask import Flask, request, render_template, jsonify
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from loguru import logger
import joblib
import numpy as np
import pandas as pd
import json
import os

app = Flask(__name__)

logger.add("app.log", rotation="10 MB", level="INFO")
logger.info("Starting Heart Disease Prediction application")

model_info = {
    "Logistic Regression": {"file": "logistic_regression_model.pkl", "label": "Logistic Regression"},
    "Naive Bayes": {"file": "naive_bayes_model.pkl", "label": "Naive Bayes"},
    "SVM": {"file": "support_vector_machine_model.pkl", "label": "Support Vector Machine"},
    "KNN": {"file": "k_nearest_neighbors_model.pkl", "label": "K-Nearest Neighbors"},
    "Decision Tree": {"file": "decision_tree_model.pkl", "label": "Decision Tree"},
    "Random Forest": {"file": "random_forest_model.pkl", "label": "Random Forest"},
    "XGBoost": {"file": "xgboost_model.pkl", "label": "XGBoost"},
    "Neural Network": {"file": "neural_network_model.pkl", "label": "Neural Network"},
    "Voting Ensemble": {"file": "voting_ensemble_model.pkl", "label": "Voting Ensemble"},
}

models = {}
for k, v in model_info.items():
    try:
        models[k] = joblib.load(f"models/{v['file']}")
        logger.info(f"Loaded model: {k}")
    except (ModuleNotFoundError, FileNotFoundError) as e:
        logger.warning(f"Could not load model '{k}': {e}")

try:
    scaler = joblib.load("models/scaler.pkl")
    logger.info("Loaded StandardScaler")
except (FileNotFoundError, Exception) as e:
    scaler = None
    logger.warning(f"Could not load scaler: {e}")

try:
    results_df = pd.read_csv("static/results/comparison.csv")
    model_accuracies = {}
    for _, row in results_df.iterrows():
        model_accuracies[row["Model"]] = {
            "accuracy": row["Accuracy"], "precision": row["Precision"],
            "recall": row["Recall"], "f1": row["F1-Score"],
            "roc_auc": row["ROC-AUC"], "specificity": row["Specificity"]
        }
    logger.info(f"Loaded metrics for {len(model_accuracies)} models")
except (FileNotFoundError, KeyError) as e:
    model_accuracies = {}
    logger.warning(f"Could not load model metrics: {e}")

feature_importance = {}
try:
    with open("static/results/feature_importance.json") as f:
        feature_importance = json.load(f)
    logger.info(f"Loaded feature importance for {len(feature_importance)} models")
except (FileNotFoundError, json.JSONDecodeError) as e:
    logger.warning(f"Could not load feature importance: {e}")

FEATURE_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_LABELS = {
    "age": "Age", "sex": "Sex", "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure", "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar", "restecg": "Resting ECG",
    "thalach": "Max Heart Rate", "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)", "slope": "ST Segment Slope",
    "ca": "Major Vessels (Fluoroscopy)", "thal": "Thalassemia"
}

FEATURE_DESCRIPTIONS = {
    "age": "Age in years",
    "sex": "1 = male, 0 = female",
    "cp": "1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic",
    "trestbps": "Resting blood pressure in mm Hg",
    "chol": "Serum cholesterol in mg/dl",
    "fbs": "Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)",
    "restecg": "0: normal, 1: ST-T wave abnormality, 2: left ventricular hypertrophy",
    "thalach": "Maximum heart rate achieved",
    "exang": "Exercise induced angina (1 = yes, 0 = no)",
    "oldpeak": "ST depression induced by exercise relative to rest",
    "slope": "0: upsloping, 1: flat, 2: downsloping",
    "ca": "Number of major vessels colored by fluoroscopy (0-4)",
    "thal": "3: normal, 6: fixed defect, 7: reversible defect"
}

class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0, le=150, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=300, description="Resting blood pressure in mm Hg")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol in mg/dl")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG results (0-2)")
    thalach: float = Field(..., ge=30, le=250, description="Maximum heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=0, le=2, description="ST segment slope (0-2)")
    ca: int = Field(..., ge=0, le=4, description="Number of major vessels colored (0-4)")
    thal: int = Field(..., ge=3, le=7, description="Thalassemia (3, 6, 7)")
    model: str = Field(..., description="Model name to use for prediction")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
            "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
            "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 3,
            "model": "Random Forest"
        }
    })

def get_top_features(model_name, n=5):
    if model_name in feature_importance:
        imp_dict = feature_importance[model_name]
        sorted_feats = sorted(imp_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        max_imp = max(abs(v) for v in imp_dict.values()) if imp_dict else 1
        return [
            {"name": name, "importance": val, "pct": round(abs(val) / max_imp * 100, 1)}
            for name, val in sorted_feats[:n]
        ]
    return []

def sort_model_key(name):
    order = ["Logistic Regression", "Naive Bayes", "SVM", "KNN",
             "Decision Tree", "Random Forest", "XGBoost", "Neural Network",
             "Voting Ensemble"]
    return order.index(name) if name in order else 999

def predict_from_features(features_list, model_name):
    features_arr = np.array(features_list).reshape(1, -1)
    if scaler is not None:
        features_arr = scaler.transform(features_arr)
    model = models[model_name]
    prediction_value = model.predict(features_arr)[0]
    result = int(prediction_value)
    raw_output = float(prediction_value)
    prediction_text = "No Heart Disease" if result == 0 else "Heart Disease"
    pred_class = "success" if result == 0 else "danger"
    acc_info = model_accuracies.get(model_name, {})
    top_features = get_top_features(model_name, 5)
    return {
        "model": model_name,
        "prediction": result,
        "raw_output": raw_output,
        "prediction_text": prediction_text,
        "prediction_class": pred_class,
        "accuracy": acc_info.get("accuracy", "N/A"),
        "precision": acc_info.get("precision", "N/A"),
        "recall": acc_info.get("recall", "N/A"),
        "f1": acc_info.get("f1", "N/A"),
        "roc_auc": acc_info.get("roc_auc", "N/A"),
        "specificity": acc_info.get("specificity", "N/A"),
        "top_features": top_features,
    }

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    form_data = {}
    if request.method == "POST":
        model_name = request.form.get("model")
        if model_name not in models:
            return jsonify({"error": f"Model '{model_name}' not available!"}), 400
        try:
            features = []
            for name in FEATURE_NAMES:
                val = request.form.get(name)
                form_data[name] = val
                if val is None or val.strip() == "":
                    return jsonify({"error": f"Missing value for {FEATURE_LABELS[name]}"}), 400
                features.append(float(val))
            prediction = predict_from_features(features, model_name)
        except ValueError:
            return jsonify({"error": "Invalid input. All values must be numeric."}), 400
    return render_template("index.html", prediction=prediction,
                           form_data=form_data,
                           feature_labels=FEATURE_LABELS,
                           feature_descriptions=FEATURE_DESCRIPTIONS,
                           feature_names=FEATURE_NAMES,
                           model_names=list(models.keys()))

@app.route("/models")
def models_page():
    try:
        df = pd.read_csv("static/results/comparison.csv")
        table_data = df.to_dict(orient="records")
        table_data.sort(key=lambda r: sort_model_key(r.get("Model", "")))
    except (FileNotFoundError, KeyError):
        table_data = []
    return render_template("models.html", table_data=table_data,
                           feature_importance_available=bool(feature_importance))

@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = PredictionRequest(**request.json)
    except ValidationError as e:
        logger.warning(f"API validation error: {e}")
        return jsonify({"error": "Validation failed", "details": e.errors()}), 400
    except Exception as e:
        logger.warning(f"API request error: {e}")
        return jsonify({"error": "Invalid request body"}), 400

    if data.model not in models:
        return jsonify({"error": f"Model '{data.model}' not available. Options: {list(models.keys())}"}), 400

    try:
        features = [
            data.age, data.sex, data.cp, data.trestbps, data.chol,
            data.fbs, data.restecg, data.thalach, data.exang,
            data.oldpeak, data.slope, data.ca, data.thal
        ]
        result = predict_from_features(features, data.model)
        logger.info(f"API prediction: {data.model} -> {result['prediction_text']}")
        return jsonify(result)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({
        "models": list(models.keys()),
        "feature_names": FEATURE_NAMES,
        "feature_labels": FEATURE_LABELS,
    })

@app.route("/api/metrics", methods=["GET"])
def api_metrics():
    return jsonify(model_accuracies)

@app.route("/api/feature_importance", methods=["GET"])
def api_feature_importance():
    return jsonify(feature_importance)

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True)
