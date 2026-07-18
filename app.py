from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

model_info = {
    "Logistic Regression": {"file": "logistic_regression_model.pkl", "label": "Logistic Regression"},
    "Naive Bayes": {"file": "naive_bayes_model.pkl", "label": "Naive Bayes"},
    "SVM": {"file": "support_vector_machine_model.pkl", "label": "Support Vector Machine"},
    "KNN": {"file": "k_nearest_neighbors_model.pkl", "label": "K-Nearest Neighbors"},
    "Decision Tree": {"file": "decision_tree_model.pkl", "label": "Decision Tree"},
    "Random Forest": {"file": "random_forest_model.pkl", "label": "Random Forest"},
    "XGBoost": {"file": "xgboost_model.pkl", "label": "XGBoost"},
    "Neural Network": {"file": "neural_network_model.pkl", "label": "Neural Network"},
}

models = {k: joblib.load(f"models/{v['file']}") for k, v in model_info.items()}

try:
    results_df = pd.read_csv("static/results/comparison.csv")
    model_accuracies = {}
    for _, row in results_df.iterrows():
        model_accuracies[row["Model"]] = {
            "accuracy": row["Accuracy"],
            "precision": row["Precision"],
            "recall": row["Recall"],
            "f1": row["F1-Score"],
            "roc_auc": row["ROC-AUC"],
            "specificity": row["Specificity"]
        }
except (FileNotFoundError, KeyError):
    model_accuracies = {}

FEATURE_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

FEATURE_LABELS = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Serum Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Max Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression (Oldpeak)",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels (Fluoroscopy)",
    "thal": "Thalassemia"
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


def sort_model_key(name):
    order = ["Logistic Regression", "Naive Bayes", "SVM", "KNN",
             "Decision Tree", "Random Forest", "XGBoost", "Neural Network"]
    return order.index(name) if name in order else 999


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

            features_arr = np.array(features).reshape(1, -1)
        except ValueError:
            return jsonify({"error": "Invalid input. All values must be numeric."}), 400

        model = models[model_name]
        prediction_value = model.predict(features_arr)[0]
        result = int(prediction_value)
        raw_output = float(prediction_value)
        prediction_text = "No Heart Disease" if result == 0 else "Heart Disease"

        pred_class = "danger" if result == 1 else "success"

        acc_info = model_accuracies.get(model_name, {})
        accuracy_display = acc_info.get("accuracy", "N/A")

        prediction = {
            "model": model_name,
            "prediction": result,
            "raw_output": raw_output,
            "prediction_text": prediction_text,
            "prediction_class": pred_class,
            "accuracy": accuracy_display,
            "precision": acc_info.get("precision", "N/A"),
            "recall": acc_info.get("recall", "N/A"),
            "f1": acc_info.get("f1", "N/A"),
            "roc_auc": acc_info.get("roc_auc", "N/A"),
        }

    return render_template("index.html", prediction=prediction,
                           form_data=form_data,
                           feature_labels=FEATURE_LABELS,
                           feature_descriptions=FEATURE_DESCRIPTIONS,
                           feature_names=FEATURE_NAMES,
                           model_names=list(model_info.keys()))


@app.route("/models")
def models_page():
    try:
        df = pd.read_csv("static/results/comparison.csv")
        table_data = df.to_dict(orient="records")
        table_data.sort(key=lambda r: sort_model_key(r.get("Model", "")))
    except (FileNotFoundError, KeyError):
        table_data = []
    return render_template("models.html", table_data=table_data)


if __name__ == "__main__":
    app.run(debug=True)
