import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import warnings
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")

print("Downloading dataset from Kaggle...")
cache_path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
os.makedirs("data", exist_ok=True)
local_csv = os.path.join("data", "heart.csv")
shutil.copy(os.path.join(cache_path, "heart.csv"), local_csv)
print(f"Dataset saved to: {os.path.abspath(local_csv)}")

dataset = pd.read_csv(local_csv)
print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
print(f"Target distribution:\n{dataset['target'].value_counts()}")

print("\nCorrelation with target:")
print(dataset.corr()["target"].abs().sort_values(ascending=False))

predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(
    predictors, target, test_size=0.20, random_state=0
)
print(f"\nTraining data: {X_train.shape}, Testing data: {X_test.shape}")

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": svm.SVC(kernel='linear'),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42)
}

model_scores = {}

os.makedirs("models", exist_ok=True)

print("Training models...")
for name, model in tqdm(models.items(), desc="Training Progress", unit="model"):
    model.fit(X_train, Y_train)
    filename = name.lower().replace(" ", "_").replace("-", "_") + "_model.pkl"
    joblib.dump(model, f"models/{filename}")
    predictions = model.predict(X_test)
    accuracy = round(accuracy_score(predictions, Y_test) * 100, 2)
    model_scores[name] = accuracy
    print(f"{name}: {accuracy}%")

print("\nTraining Neural Network (MLPClassifier)...")
nn = MLPClassifier(
    hidden_layer_sizes=(11,),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
nn.fit(X_train, Y_train)
joblib.dump(nn, "models/neural_network_model.pkl")
predictions_nn = nn.predict(X_test)
accuracy_nn = round(accuracy_score(predictions_nn, Y_test) * 100, 2)
model_scores["Neural Network"] = accuracy_nn
print(f"Neural Network: {accuracy_nn}%")

print("\nFinal Results:")
for name, score in model_scores.items():
    print(f"{name}: {score}%")

algorithms = list(model_scores.keys())
scores = list(model_scores.values())

sns.set(rc={'figure.figsize': (15, 8)})
sns.barplot(x=algorithms, y=scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score")
plt.title("Model Performance Comparison")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Training complete! Models saved to models/")
