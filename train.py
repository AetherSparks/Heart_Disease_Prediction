from loguru import logger
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
import json
import warnings
import kagglehub
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import joblib
from tqdm import tqdm

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

FEATURE_NAMES = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
                 "thalach", "exang", "oldpeak", "slope", "ca", "thal"]

logger.add("training.log", rotation="10 MB", level="INFO")
logger.info("=== Heart Disease Prediction — Enhanced Training Pipeline ===")

logger.info("Downloading dataset from Kaggle...")
cache_path = kagglehub.dataset_download("johnsmith88/heart-disease-dataset")
os.makedirs("data", exist_ok=True)
local_csv = os.path.join("data", "heart.csv")
shutil.copy(os.path.join(cache_path, "heart.csv"), local_csv)
logger.info(f"Dataset saved to: {os.path.abspath(local_csv)}")

dataset = pd.read_csv(local_csv)
logger.info(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")

dupes = dataset.duplicated(keep=False).sum()
if dupes:
    dataset = dataset.drop_duplicates()
    logger.info(f"Removed {dupes} duplicate rows ({dataset.shape[0]} unique rows remaining)")

logger.info(f"Target distribution:\n{dataset['target'].value_counts().to_string()}")

predictors = dataset.drop("target", axis=1)
target = dataset["target"]
X_train, X_test, Y_train, Y_test = train_test_split(
    predictors, target, test_size=0.20, random_state=0, stratify=target
)
logger.info(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

logger.info("Applying SMOTE to balance training classes...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

smote = SMOTE(random_state=42)
X_train_resampled, Y_train_resampled = smote.fit_resample(X_train_scaled, Y_train)
logger.info(f"After SMOTE — Train: {X_train_resampled.shape[0]} samples")
logger.info(f"Resampled target distribution:\n{pd.Series(Y_train_resampled).value_counts().to_string()}")

os.makedirs("models", exist_ok=True)
os.makedirs("static/results", exist_ok=True)

def train_with_gridsearch(model, param_grid, X, Y, name, cv=5):
    logger.info(f"Tuning hyperparameters for {name}...")
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    gs = GridSearchCV(
        model, param_grid, cv=skf, scoring='accuracy',
        n_jobs=-1, verbose=0
    )
    gs.fit(X, Y)
    logger.info(f"  Best params: {gs.best_params_}")
    logger.info(f"  Best CV accuracy: {gs.best_score_*100:.2f}%")
    return gs.best_estimator_

param_grids = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "params": {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
    },
    "Naive Bayes": {
        "model": GaussianNB(),
        "params": {}
    },
    "Support Vector Machine": {
        "model": svm.SVC(probability=True, random_state=42),
        "params": {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", "auto"]}
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {"n_neighbors": [3, 5, 7, 9, 11, 15], "weights": ["uniform", "distance"]}
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=42),
        "params": {"max_depth": [3, 5, 7, 10, None], "min_samples_split": [2, 5, 10]}
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {"n_estimators": [50, 100, 200], "max_depth": [5, 10, None], "min_samples_split": [2, 5]}
    },
    "XGBoost": {
        "model": xgb.XGBClassifier(objective="binary:logistic", random_state=42, verbosity=0),
        "params": {"n_estimators": [50, 100], "max_depth": [3, 5, 7], "learning_rate": [0.01, 0.1, 0.3]}
    },
}

models = {}
results_rows = []
feature_importance_data = {}

logger.info("=== Training Models with Hyperparameter Tuning ===")
for name, cfg in tqdm(param_grids.items(), desc="Training", unit="model"):
    if cfg["params"]:
        clf = train_with_gridsearch(cfg["model"], cfg["params"], X_train_resampled, Y_train_resampled, name)
    else:
        logger.info(f"Training {name} (no hyperparameter tuning)...")
        clf = cfg["model"].fit(X_train_resampled, Y_train_resampled)
    models[name] = clf
    filename = name.lower().replace(" ", "_").replace("-", "_") + "_model.pkl"
    joblib.dump(clf, f"models/{filename}")

    Y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(Y_test, Y_pred)
    prec = precision_score(Y_test, Y_pred)
    rec = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)
    try:
        Y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        roc_auc = roc_auc_score(Y_test, Y_prob)
    except Exception:
        roc_auc = float('nan')
    cm = confusion_matrix(Y_test, Y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float('nan')

    results_rows.append({
        "Model": name, "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2), "Recall": round(rec * 100, 2),
        "F1-Score": round(f1 * 100, 2),
        "ROC-AUC": round(roc_auc * 100, 2) if not np.isnan(roc_auc) else "N/A",
        "Specificity": round(specificity * 100, 2) if not np.isnan(specificity) else "N/A",
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)
    })
    logger.info(f"{name}: Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%  F1={f1*100:.2f}%  ROC-AUC={roc_auc*100:.2f}%")

    logger.info(f"Computing permutation feature importance for {name}...")
    try:
        perm_result = permutation_importance(
            clf, X_test_scaled, Y_test, n_repeats=10,
            random_state=42, scoring='accuracy', n_jobs=-1
        )
        importance_dict = {}
        for i, fname in enumerate(FEATURE_NAMES):
            importance_dict[fname] = round(perm_result.importances_mean[i], 4)
        feature_importance_data[name] = importance_dict
    except Exception as e:
        logger.warning(f"Permutation importance failed for {name}: {e}")
        try:
            if hasattr(clf, 'feature_importances_'):
                importances = clf.feature_importances_
                feature_importance_data[name] = {
                    fname: round(float(importances[i]), 4)
                    for i, fname in enumerate(FEATURE_NAMES)
                }
            elif hasattr(clf, 'coef_'):
                coefs = np.abs(clf.coef_[0])
                feature_importance_data[name] = {
                    fname: round(float(coefs[i]), 4)
                    for i, fname in enumerate(FEATURE_NAMES)
                }
            else:
                feature_importance_data[name] = {f: 0.0 for f in FEATURE_NAMES}
        except Exception:
            feature_importance_data[name] = {f: 0.0 for f in FEATURE_NAMES}

logger.info("=== Training Neural Network (MLPClassifier) ===")
nn_param_grid = {
    "hidden_layer_sizes": [(11,), (20,), (11, 5)],
    "activation": ["relu", "tanh"],
    "alpha": [0.0001, 0.001]
}
nn = train_with_gridsearch(
    MLPClassifier(solver='adam', max_iter=500, random_state=42),
    nn_param_grid, X_train_resampled, Y_train_resampled, "Neural Network"
)
models["Neural Network"] = nn
joblib.dump(nn, "models/neural_network_model.pkl")

Y_pred_nn = nn.predict(X_test_scaled)
acc_nn = accuracy_score(Y_test, Y_pred_nn)
prec_nn = precision_score(Y_test, Y_pred_nn)
rec_nn = recall_score(Y_test, Y_pred_nn)
f1_nn = f1_score(Y_test, Y_pred_nn)
roc_auc_nn = roc_auc_score(Y_test, nn.predict_proba(X_test_scaled)[:, 1])
cm_nn = confusion_matrix(Y_test, Y_pred_nn)
tn_nn, fp_nn, fn_nn, tp_nn = cm_nn.ravel()
spec_nn = tn_nn / (tn_nn + fp_nn) if (tn_nn + fp_nn) > 0 else float('nan')
results_rows.append({
    "Model": "Neural Network", "Accuracy": round(acc_nn * 100, 2),
    "Precision": round(prec_nn * 100, 2), "Recall": round(rec_nn * 100, 2),
    "F1-Score": round(f1_nn * 100, 2), "ROC-AUC": round(roc_auc_nn * 100, 2),
    "Specificity": round(spec_nn * 100, 2),
    "TN": int(tn_nn), "FP": int(fp_nn), "FN": int(fn_nn), "TP": int(tp_nn)
})
logger.info(f"Neural Network: Acc={acc_nn*100:.2f}%  Prec={prec_nn*100:.2f}%  Rec={rec_nn*100:.2f}%  F1={f1_nn*100:.2f}%  ROC-AUC={roc_auc_nn*100:.2f}%")

logger.info("Computing permutation feature importance for Neural Network...")
try:
    perm_result = permutation_importance(
        nn, X_test_scaled, Y_test, n_repeats=10,
        random_state=42, scoring='accuracy', n_jobs=-1
    )
    feature_importance_data["Neural Network"] = {
        FEATURE_NAMES[i]: round(perm_result.importances_mean[i], 4)
        for i in range(len(FEATURE_NAMES))
    }
except Exception as e:
    logger.warning(f"Permutation importance failed for Neural Network: {e}")
    feature_importance_data["Neural Network"] = {f: 0.0 for f in FEATURE_NAMES}

logger.info("=== Training VotingClassifier Ensemble (soft voting) ===")
ensemble_model_names = ["Logistic Regression", "Random Forest", "XGBoost", "Neural Network"]
ensemble_estimators = [(name.replace(" ", "_").lower(), models[name]) for name in ensemble_model_names]
voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft')
voting_clf.fit(X_train_resampled, Y_train_resampled)
joblib.dump(voting_clf, "models/voting_ensemble_model.pkl")
models["Voting Ensemble"] = voting_clf

Y_pred_ens = voting_clf.predict(X_test_scaled)
acc_ens = accuracy_score(Y_test, Y_pred_ens)
prec_ens = precision_score(Y_test, Y_pred_ens)
rec_ens = recall_score(Y_test, Y_pred_ens)
f1_ens = f1_score(Y_test, Y_pred_ens)
roc_auc_ens = roc_auc_score(Y_test, voting_clf.predict_proba(X_test_scaled)[:, 1])
cm_ens = confusion_matrix(Y_test, Y_pred_ens)
tn_ens, fp_ens, fn_ens, tp_ens = cm_ens.ravel()
spec_ens = tn_ens / (tn_ens + fp_ens) if (tn_ens + fp_ens) > 0 else float('nan')
results_rows.append({
    "Model": "Voting Ensemble", "Accuracy": round(acc_ens * 100, 2),
    "Precision": round(prec_ens * 100, 2), "Recall": round(rec_ens * 100, 2),
    "F1-Score": round(f1_ens * 100, 2), "ROC-AUC": round(roc_auc_ens * 100, 2),
    "Specificity": round(spec_ens * 100, 2),
    "TN": int(tn_ens), "FP": int(fp_ens), "FN": int(fn_ens), "TP": int(tp_ens)
})
logger.info(f"Voting Ensemble: Acc={acc_ens*100:.2f}%  Prec={prec_ens*100:.2f}%  Rec={rec_ens*100:.2f}%  F1={f1_ens*100:.2f}%  ROC-AUC={roc_auc_ens*100:.2f}%")

results_df = pd.DataFrame(results_rows)
results_df.to_csv("static/results/comparison.csv", index=False)
logger.info(f"Results saved to static/results/comparison.csv")

with open("static/results/feature_importance.json", "w") as f:
    json.dump(feature_importance_data, f, indent=2)
logger.info("Feature importance saved to static/results/feature_importance.json")

joblib.dump(scaler, "models/scaler.pkl")
logger.info("StandardScaler saved to models/scaler.pkl")

metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Specificity"]
numeric_df = results_df.copy()
for c in metric_cols:
    numeric_df[c] = pd.to_numeric(numeric_df[c], errors='coerce')

logger.info("Generating visualizations...")

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

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
axes = axes.flatten()
sorted_models = sorted(feature_importance_data.keys())
for idx, model_name in enumerate(sorted_models):
    if idx >= len(axes):
        break
    ax = axes[idx]
    imp_dict = feature_importance_data[model_name]
    sorted_features = sorted(imp_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    feature_names, importances = zip(*sorted_features)
    colors = ['#ef4444' if v > 0 else '#3b82f6' for v in importances]
    ax.barh(range(len(feature_names)), importances, color=colors, alpha=0.8)
    ax.set_yticks(range(len(feature_names)))
    ax.set_yticklabels(feature_names, fontsize=9)
    ax.set_xlabel("Importance", fontsize=9)
    ax.set_title(model_name, fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='gray', linewidth=0.5, linestyle='--')
    ax.tick_params(axis='both', labelsize=8)
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])
plt.suptitle("Feature Importance by Model (Permutation)", fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("static/results/feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()
logger.info("Feature importance chart saved to static/results/feature_importance.png")

logger.info("\n" + "=" * 70)
logger.info("FINAL RESULTS")
logger.info("=" * 70)
logger.info("\n" + results_df.to_string(index=False))
logger.info("=" * 70)
logger.info("Training complete! Models saved to models/, results saved to static/results/")
