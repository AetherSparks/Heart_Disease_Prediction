from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
    "SVM": joblib.load("models/support_vector_machine_model.pkl"),
    "KNN": joblib.load("models/k_nearest_neighbors_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "XGBoost": joblib.load("models/xgboost_model.pkl"),
    "Neural Network": joblib.load("models/neural_network_model.pkl"),
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        input_features = request.form.get("features")
        if not input_features:
            return jsonify({"error": "No input features provided!"}), 400

        try:
            features = np.array([float(x) for x in input_features.split(",")])
            if len(features) != 13:
                return jsonify({"error": "Input features should be 13 values!"}), 400
            features = features.reshape(1, -1)
        except ValueError:
            return jsonify({"error": "Invalid input format. Provide comma-separated numeric values!"}), 400

        model_name = request.form.get("model")
        if model_name not in models:
            return jsonify({"error": f"Model '{model_name}' not available!"}), 400

        model = models[model_name]

        prediction_value = model.predict(features)[0]
        result = int(prediction_value)
        raw_output = float(prediction_value)

        prediction_text = "No Heart Disease" if result == 0 else "Heart Disease"

        prediction = {
            "model": model_name,
            "prediction": result,
            "raw_output": raw_output,
            "prediction_text": prediction_text
        }

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
