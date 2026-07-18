import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import warnings
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
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
    "Support Vector Machine": svm.SVC(kernel='linear', probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": xgb.XGBClassifier(objective="binary:logistic", random_state=42)
}

os.makedirs("models", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

results_rows = []

print("Training models...")
all_models = list(models.items())

for name, clf in tqdm(all_models, desc="Training Progress", unit="model"):
    clf.fit(X_train, Y_train)
    filename = name.lower().replace(" ", "_").replace("-", "_") + "_model.pkl"
    joblib.dump(clf, f"models/{filename}")

    Y_pred = clf.predict(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    try:
        Y_prob = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
    except Exception:
        roc_auc = float('nan')

    cm = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    results_rows.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "ROC-AUC": round(roc_auc * 100, 2) if not np.isnan(roc_auc) else "N/A",
        "Specificity": round(specificity * 100, 2) if not np.isnan(specificity) else "N/A",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    })

    print(f"{name}: Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  F1={f1*100:.2f}%  ROC-AUC={roc_auc*100:.2f}%")

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

Y_pred_nn = nn.predict(X_test)
Y_prob_nn = nn.predict_proba(X_test)[:, 1]

acc_nn = accuracy_score(Y_test, Y_pred_nn)
prec_nn = precision_score(Y_test, Y_pred_nn)
rec_nn = recall_score(Y_test, Y_pred_nn)
f1_nn = f1_score(Y_test, Y_pred_nn)
roc_auc_nn = roc_auc_score(Y_test, Y_prob_nn)
cm_nn = confusion_matrix(Y_test, Y_pred_nn)
tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()
spec_nn = tn_nn / (tn_nn + fp_nn) if (tn_nn + fp_nn) > 0 else float('nan')

results_rows.append({
    "Model": "Neural Network",
    "Accuracy": round(acc_nn * 100, 2),
    "Precision": round(prec_nn * 100, 2),
    "Recall": round(rec_nn * 100, 2),
    "F1-Score": round(f1_nn * 100, 2),
    "ROC-AUC": round(roc_auc_nn * 100, 2),
    "Specificity": round(spec_nn * 100, 2),
    "TN": int(tn_nn), "FP": int(fp_nn), "FN": int(fn_nn), "TP": int(tp_nn)
})

print(f"Neural Network: Acc={acc_nn*100:.2f}%  Prec={prec_nn*100:.2f}%  Rec={rec_nn*100:.2f}%  F1={f1_nn*100:.2f}%  ROC-AUC={roc_auc_nn*100:.2f}%")

results_df = pd.DataFrame(results_rows)
results_csv = "static/results/comparison.csv"
results_df.to_csv(results_csv, index=False)
print(f"\nResults saved to {results_csv}")

metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Specificity"]

numeric_df = results_df.copy()
for c in metric_cols:
    numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce')

sns.set_theme(style="whitegrid")

fig, ax = plt.subplots(figsize=(14, 7))
x = np.arange(len(metric_cols))
for i, row in numeric_df.iterrows():
    values = [row[c] if pd.notna(row[c]) else 0 for c in metric_cols]
    ax.plot(metric_cols, values, marker='o', linewidth=2, label=row["Model"])

ax.set_xlabel("Metric", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison Across Metrics", fontsize=14)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.set_ylim(50, 105)
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("static/results/comparison_line.png", dpi=150, bbox_inches='tight')
plt.close()
print("Line chart saved to static/results/comparison_line.png")

fig, ax = plt.subplots(figsize=(14, 7))
bar_width = 0.1
x = np.arange(len(numeric_df))
for j, metric in enumerate(metric_cols):
    vals = numeric_df[metric].fillna(0)
    bars = ax.bar(x + j * bar_width, vals, bar_width, label=metric)
    for bar, v in zip(bars, vals):
        if pd.notna(v) and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{v:.1f}', ha='center', va='bottom', fontsize=7)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score (%)", fontsize=12)
ax.set_title("Model Performance Comparison (Grouped Bar)", fontsize=14)
ax.set_xticks(x + bar_width * (len(metric_cols) - 1) / 2)
ax.set_xticklabels(numeric_df["Model"], rotation=30, ha='right')
ax.legend(loc='lower right')
ax.set_ylim(0, 110)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("static/results/comparison_bar.png", dpi=150, bbox_inches='tight')
plt.close()
print("Bar chart saved to static/results/comparison_bar.png")

n_models = len(numeric_df)
cols = 4
rows = int(np.ceil(n_models / cols))
fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
axes = axes.flatten()

for i, (_, row) in enumerate(numeric_df.iterrows()):
    ax_i = axes[i]
    tn, fp, fn, tp = int(row["TN"]), int(row["FP"]), int(row["FN"]), int(row["TP"])
    cm_data = np.array([[tn, fp], [fn, tp]])
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'], ax=ax_i)
    ax_i.set_title(row["Model"], fontsize=10)
    ax_i.set_xlabel("Predicted")
    ax_i.set_ylabel("Actual")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("static/results/confusion_matrices.png", dpi=150, bbox_inches='tight')
plt.close()
print("Confusion matrices saved to static/results/confusion_matrices.png")

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(results_df.to_string(index=False))
print("=" * 70)

print("\nTraining complete! Models saved to models/, results saved to static/results/")
