"""
Predictive Modeling Using Machine Learning
==========================================
Project: Build a model to predict outcomes based on given data.
Features:
  - Linear Regression, Decision Trees, Random Forest
  - Train/test split and accuracy evaluation
  - Confusion matrix and ROC curve visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc, ConfusionMatrixDisplay
)

# ── 1. Load Dataset ──────────────────────────────────────────────────────────
print("=" * 60)
print("  PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 60)
print("\n[1] Loading Dataset: Breast Cancer Wisconsin\n")

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print(f"    Samples  : {X.shape[0]}")
print(f"    Features : {X.shape[1]}")
print(f"    Classes  : {list(data.target_names)}  (0=Malignant, 1=Benign)")
print(f"    Class Distribution:\n{y.value_counts().rename({0:'Malignant', 1:'Benign'}).to_string()}")

# ── 2. Preprocessing ─────────────────────────────────────────────────────────
print("\n[2] Preprocessing: Train/Test Split (80/20) + Feature Scaling\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}")

# ── 3. Train Models ───────────────────────────────────────────────────────────
print("\n[3] Training Models...\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"      : DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest"      : RandomForestClassifier(n_estimators=100, random_state=42),
}

results = {}
for name, model in models.items():
    # Use scaled data for Logistic Regression, raw for tree models
    Xtr = X_train_sc if "Logistic" in name else X_train
    Xte = X_test_sc  if "Logistic" in name else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    X_cv = X_train_sc if "Logistic" in name else X_train
    cv  = cross_val_score(model, X_cv, y_train,
                          cv=5, scoring='accuracy').mean()
    cm  = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    results[name] = {
        "model": model, "y_pred": y_pred, "y_prob": y_prob,
        "acc": acc, "cv_acc": cv, "cm": cm,
        "fpr": fpr, "tpr": tpr, "auc": roc_auc,
        "report": classification_report(y_test, y_pred,
                                        target_names=data.target_names)
    }
    print(f"    {name:<25} Test Acc: {acc:.4f}   CV Acc: {cv:.4f}   AUC: {roc_auc:.4f}")

# ── 4. Detailed Classification Reports ───────────────────────────────────────
print("\n[4] Classification Reports\n")
for name, r in results.items():
    print(f"  ── {name} ──")
    print(r["report"])

# ── 5. Visualisations ────────────────────────────────────────────────────────
print("\n[5] Generating Visualisations...\n")

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#4C72B0', '#DD8452', '#55A868']
fig = plt.figure(figsize=(20, 22))
fig.patch.set_facecolor('#F8F9FA')
gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

model_names = list(results.keys())

# ── 5a. Accuracy Comparison Bar ──────────────────────────────────────────────
ax_acc = fig.add_subplot(gs[0, :])
accs   = [results[n]["acc"]    for n in model_names]
cv_acc = [results[n]["cv_acc"] for n in model_names]
x = np.arange(len(model_names))
bars1 = ax_acc.bar(x - 0.2, accs,   0.38, label='Test Accuracy',  color=COLORS, alpha=0.85)
bars2 = ax_acc.bar(x + 0.2, cv_acc, 0.38, label='CV Accuracy (5-fold)',
                   color=COLORS, alpha=0.45, edgecolor='black', linewidth=0.8)
for bar in bars1:
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=11, fontweight='bold')
for bar in bars2:
    ax_acc.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=10, color='#333')
ax_acc.set_xticks(x); ax_acc.set_xticklabels(model_names, fontsize=12)
ax_acc.set_ylim(0.88, 1.02)
ax_acc.set_title("Model Accuracy Comparison", fontsize=15, fontweight='bold', pad=12)
ax_acc.set_ylabel("Accuracy", fontsize=12)
ax_acc.legend(fontsize=11)

# ── 5b. Confusion Matrices ───────────────────────────────────────────────────
for i, (name, r) in enumerate(results.items()):
    ax = fig.add_subplot(gs[1, i])
    disp = ConfusionMatrixDisplay(confusion_matrix=r["cm"],
                                  display_labels=data.target_names)
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Confusion Matrix\n{name}", fontsize=11, fontweight='bold')
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual",    fontsize=10)

# ── 5c. ROC Curves ───────────────────────────────────────────────────────────
ax_roc = fig.add_subplot(gs[2, :2])
for i, (name, r) in enumerate(results.items()):
    ax_roc.plot(r["fpr"], r["tpr"], color=COLORS[i], lw=2.2,
                label=f"{name}  (AUC = {r['auc']:.3f})")
ax_roc.plot([0,1],[0,1],'k--', lw=1.2, label='Random Classifier')
ax_roc.fill_between([0,1],[0,1], alpha=0.05, color='grey')
ax_roc.set_xlabel("False Positive Rate", fontsize=12)
ax_roc.set_ylabel("True Positive Rate",  fontsize=12)
ax_roc.set_title("ROC Curves – All Models", fontsize=14, fontweight='bold')
ax_roc.legend(fontsize=11, loc='lower right')

# ── 5d. AUC Bar ──────────────────────────────────────────────────────────────
ax_auc = fig.add_subplot(gs[2, 2])
aucs = [results[n]["auc"] for n in model_names]
bars = ax_auc.barh(model_names, aucs, color=COLORS, alpha=0.85)
for bar, val in zip(bars, aucs):
    ax_auc.text(val - 0.005, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va='center', ha='right', fontsize=11,
                fontweight='bold', color='white')
ax_auc.set_xlim(0.95, 1.01)
ax_auc.set_title("AUC Scores", fontsize=13, fontweight='bold')
ax_auc.set_xlabel("AUC", fontsize=11)

# ── 5e. Feature Importance (Random Forest) ────────────────────────────────────
ax_fi = fig.add_subplot(gs[3, :])
rf_model = results["Random Forest"]["model"]
importances = pd.Series(rf_model.feature_importances_, index=data.feature_names)
top15 = importances.nlargest(15).sort_values()
colors_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, 15))
bars_fi = ax_fi.barh(top15.index, top15.values, color=colors_fi, edgecolor='white', linewidth=0.5)
for bar, val in zip(bars_fi, top15.values):
    ax_fi.text(val + 0.001, bar.get_y() + bar.get_height()/2,
               f"{val:.4f}", va='center', fontsize=9)
ax_fi.set_title("Top 15 Feature Importances – Random Forest", fontsize=13, fontweight='bold')
ax_fi.set_xlabel("Importance Score", fontsize=11)

# ── Title & Save ─────────────────────────────────────────────────────────────
fig.suptitle("Predictive Modeling Using Machine Learning\nBreast Cancer Classification",
             fontsize=17, fontweight='bold', y=0.98, color='#1a1a2e')

import os
save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_predictive_modeling_report.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print("    Saved → ml_predictive_modeling_report.png")

# ── 6. Summary ────────────────────────────────────────────────────────────────
best = max(results, key=lambda n: results[n]["auc"])
print("\n" + "=" * 60)
print("  SUMMARY")
print("=" * 60)
print(f"  Best Model  : {best}")
print(f"  Test Acc    : {results[best]['acc']:.4f}")
print(f"  AUC Score   : {results[best]['auc']:.4f}")
print("\n  All models trained, evaluated, and visualised successfully.")
print("=" * 60)
