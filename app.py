from keras.models import load_model
from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np


app = Flask(__name__)

# Load models
models = {
    "Logistic Regression": joblib.load("models/logistic_regression_model.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl"),
    "SVM": joblib.load("models/support_vector_machine_model.pkl"),
    "KNN": joblib.load("models/k-nearest_neighbors_model.pkl"),
    "Decision Tree": joblib.load("models/decision_tree_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "XGBoost": joblib.load("models/xgboost_model.pkl"),
    "Neural Network": load_model("models/neural_network_model.h5"),
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get input features
        input_features = request.form.get("features")
        if not input_features:
            return jsonify({"error": "No input features provided!"}), 400

        try:
            # Convert input string to a NumPy array
            features = np.array([float(x) for x in input_features.split(",")])

            # Ensure the correct number of features (13)
            if len(features) != 13:
                return jsonify({"error": "Input features should be 13 values!"}), 400

            features = features.reshape(1, -1)
        except ValueError:
            return jsonify({"error": "Invalid input format. Provide comma-separated numeric values!"}), 400

        # Get the selected model
        model_name = request.form.get("model")
        if model_name not in models:
            return jsonify({"error": f"Model '{model_name}' not available!"}), 400

        model = models[model_name]

        # Predict using the chosen model
        if model_name == "Neural Network":
            prediction_value = model.predict(features)[0][0]
            result = round(prediction_value)
        else:
            prediction_value = model.predict(features)[0]
            result = int(prediction_value)

        # Interpretation of prediction
        prediction_text = "No Heart Disease" if result == 0 else "Heart Disease"

        prediction = {
            "model": model_name,
            "prediction": result,
            "raw_output": float(prediction_value),
            "prediction_text": prediction_text
        }

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)