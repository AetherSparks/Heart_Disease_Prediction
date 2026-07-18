# Heart Disease Prediction

A Flask web application that predicts the presence of heart disease in a patient using 8 different machine learning models. Users can input 13 clinical features via a web UI and get predictions from any model they choose.

## Tech Stack

- **Backend:** Python 3.9+, Flask, Gunicorn
- **ML Libraries:** scikit-learn, XGBoost
- **Frontend:** HTML, Bootstrap 4
- **Dataset:** [UCI Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (1025 records, downloaded via kagglehub)

## Dataset

The dataset is automatically downloaded from Kaggle via `kagglehub` when you run `train.py`. It contains **1025 patient records** with 13 clinical features and a binary target:

| # | Feature | Description |
|---|---------|-------------|
| 1 | `age` | Age in years |
| 2 | `sex` | 1 = male, 0 = female |
| 3 | `cp` | Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic) |
| 4 | `trestbps` | Resting blood pressure (mm Hg) |
| 5 | `chol` | Serum cholesterol (mg/dl) |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false) |
| 7 | `restecg` | Resting ECG results (0, 1, 2) |
| 8 | `thalach` | Maximum heart rate achieved |
| 9 | `exang` | Exercise induced angina (1 = yes, 0 = no) |
| 10 | `oldpeak` | ST depression induced by exercise relative to rest |
| 11 | `slope` | Slope of the peak exercise ST segment |
| 12 | `ca` | Number of major vessels (0-3) colored by fluoroscopy |
| 13 | `thal` | Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect) |
| **Target** | `target` | 0 = no heart disease, 1 = heart disease present |

## Models

All 8 models are trained on an 80/20 train-test split and saved to `models/`:

| Model | Type | File |
|-------|------|------|
| Logistic Regression | Linear classifier | `logistic_regression_model.pkl` |
| Naive Bayes (Gaussian) | Probabilistic classifier | `naive_bayes_model.pkl` |
| Support Vector Machine | Linear SVM | `support_vector_machine_model.pkl` |
| K-Nearest Neighbors | Distance-based (k=7) | `k_nearest_neighbors_model.pkl` |
| Decision Tree | Tree-based | `decision_tree_model.pkl` |
| Random Forest | Ensemble (bagging) | `random_forest_model.pkl` |
| XGBoost | Gradient boosting | `xgboost_model.pkl` |
| Neural Network (MLP) | 1 hidden layer (11 neurons), ReLU activation | `neural_network_model.pkl` |

## Project Structure

```
HeartDiseasePrediction/
├── app.py                    # Flask web application
├── train.py                  # Model training script
├── requirements.txt          # Python dependencies
├── vercel.json               # Vercel deployment config
├── .python-version           # Python version for Vercel
├── .gitignore
├── README.md
├── templates/
│   └── index.html            # Web UI template
├── models/                   # Pre-trained model files (8 models)
├── data/                     # Dataset (downloaded by train.py, gitignored)
└── flaskvenv/                # Virtual environment (gitignored)
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### 1. Clone & Setup

```bash
git clone https://github.com/AetherSparks/Heart_Disease_Prediction.git
cd Heart_Disease_Prediction
```

Create and activate a virtual environment:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Train Models

This downloads the dataset (1025 records) via kagglehub, trains all 8 models, and saves them to `models/`:

```bash
python train.py
```

A bar chart comparing model accuracies will be displayed at the end.

### 4. Run the Web App

```bash
python app.py
```

Open http://127.0.0.1:5000 in your browser. Enter 13 comma-separated feature values, select a model, and click **Predict**.

### Example Input

```
63,1,1,145,233,1,2,150,0,2.3,3,0,6
```

## Deploy to Vercel

This project is pre-configured for Vercel deployment:

1. Push the repo to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. Import your GitHub repository
4. Vercel auto-detects Flask — no configuration needed
5. Deploy

The `vercel.json` file and `.python-version` are already set up.

> **Note:** Vercel has a 500MB bundle size limit. The minimal `requirements.txt` (no TensorFlow/PyTorch) keeps the deployment well under this limit.
