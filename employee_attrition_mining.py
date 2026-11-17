"""
DMBI CA2 Mini Project: Employee Attrition Prediction Using Data Mining Techniques
Author: Prithviraj Virwani | Class: D15B | Roll No: 70
Dataset: IBM HR Analytics Employee Attrition & Performance Dataset (1,470 records, 35 features)
GitHub Raw Dataset URL:
https://raw.githubusercontent.com/Prithvi696969/prithvi-dmbi-ca2-employee-attrition/main/WA_Fn-UseC_-HR-Employee-Attrition.csv
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# ==============================================================================
# 1. DATA ACQUISITION (Loads directly from public GitHub repository)
# ==============================================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/Prithvi696969/prithvi-dmbi-ca2-employee-attrition/"
    "main/WA_Fn-UseC_-HR-Employee-Attrition.csv"
)

print("Loading IBM HR Employee Attrition Dataset from GitHub...")
df = pd.read_csv(DATASET_URL)
print(f"Dataset Loaded Successfully! Shape: {df.shape}")
print("\nTarget Variable Distribution (Attrition):")
print(df["Attrition"].value_counts())
print(df["Attrition"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# ==============================================================================
# 2. DATA PREPROCESSING & CLEANING
# ==============================================================================
print("\n--- Step 2: Data Preprocessing ---")
print("Missing Values Count:", df.isnull().sum().sum())
print("Duplicate Rows Count:", df.duplicated().sum())

# Drop constant / zero-variance and identifier columns
drop_cols = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]
df_clean = df.drop(columns=drop_cols)
print(f"Dropped constant/ID columns: {drop_cols}")
print(f"Cleaned Dataset Shape: {df_clean.shape}")

# Binary encoding for target and binary categorical variables
df_clean["Attrition"] = df_clean["Attrition"].map({"Yes": 1, "No": 0})
df_clean["OverTime"] = df_clean["OverTime"].map({"Yes": 1, "No": 0})
df_clean["Gender"] = df_clean["Gender"].map({"Male": 1, "Female": 0})

# One-Hot Encoding for multi-class nominal features
categorical_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "JobRole",
    "MaritalStatus",
]
df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)
print(f"Encoded Feature Matrix Shape: {df_encoded.shape}")

# Feature-Target Separation
X = df_encoded.drop("Attrition", axis=1)
y = df_encoded["Attrition"]

# Stratified 80/20 Train-Test Split (preserves 16.1% minority class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Training Set: {X_train.shape[0]} samples | Testing Set: {X_test.shape[0]} samples")

# Feature Scaling (StandardScaler for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# 3. DATA MINING ALGORITHMS IMPLEMENTATION
# ==============================================================================
models = {
    "Decision Tree (CART)": (
        DecisionTreeClassifier(
            criterion="gini", max_depth=5, class_weight="balanced", random_state=42
        ),
        False,
    ),
    "Random Forest (Ensemble)": (
        RandomForestClassifier(
            n_estimators=200, max_depth=8, class_weight="balanced", random_state=42
        ),
        False,
    ),
    "Logistic Regression": (
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        True,
    ),
}

results = []
roc_data = {}

print("\n--- Step 3 & 4: Model Training & Evaluation ---")
for name, (model, use_scaled) in models.items():
    Xtr = X_train_scaled if use_scaled else X_train
    Xte = X_test_scaled if use_scaled else X_test

    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results.append(
        {
            "Algorithm": name,
            "Accuracy (%)": round(acc * 100, 2),
            "Precision (%)": round(prec * 100, 2),
            "Recall (%)": round(rec * 100, 2),
            "F1-Score (%)": round(f1 * 100, 2),
            "ROC-AUC": round(auc, 3),
        }
    )
    roc_data[name] = (roc_curve(y_test, y_prob), auc)

    print(f"\n>>> {name}")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=["Stayed (0)", "Left (1)"]))

results_df = pd.DataFrame(results)
print("\n=== MODEL COMPARISON SUMMARY ===")
print(results_df.to_string(index=False))

# Extract Interpretable Decision Tree Rules
dt_model = models["Decision Tree (CART)"][0]
print("\n=== EXTRACTED DECISION TREE RULES (Top 3 Levels) ===")
print(export_text(dt_model, feature_names=list(X.columns), max_depth=3))

# Extract Top 10 Feature Importances from Random Forest
rf_model = models["Random Forest (Ensemble)"][0]
importances = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("\n=== TOP 10 ATTRITION DRIVERS (RANDOM FOREST FEATURE IMPORTANCE) ===")
print(importances.head(10).round(4))

# ==============================================================================
# 4. VISUALIZATIONS (Saves PNG figures for Report & Presentation)
# ==============================================================================
sns.set_style("whitegrid")

# Figure 1: Attrition Distribution & OverTime Impact
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
df["Attrition"].value_counts().plot.pie(
    autopct="%1.1f%%", colors=["#2563eb", "#dc2626"], ax=axes[0], startangle=90
)
axes[0].set_title("Overall Employee Attrition Distribution (N=1,470)")
axes[0].set_ylabel("")

sns.countplot(data=df, x="OverTime", hue="Attrition", palette=["#2563eb", "#dc2626"], ax=axes[1])
axes[1].set_title("Attrition Count by OverTime Status")
plt.tight_layout()
plt.savefig("fig1_attrition_overtime.png", dpi=150)

# Figure 2: Top 8 Feature Importances (Random Forest)
plt.figure(figsize=(9, 5))
importances.head(8).sort_values().plot.barh(color="#4f46e5")
plt.title("Top 8 Predictors of Employee Attrition (Random Forest)")
plt.xlabel("Relative Importance Score (Gini)")
plt.tight_layout()
plt.savefig("fig2_feature_importance.png", dpi=150)

# Figure 3: ROC Curve Comparison
plt.figure(figsize=(8, 6))
for name, ((fpr, tpr, _), auc_val) in roc_data.items():
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc_val:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random Baseline (AUC = 0.50)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate (Recall)")
plt.title("ROC Curve Comparison Across Data Mining Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig3_roc_comparison.png", dpi=150)

# Figure 4: Confusion Matrix Heatmap (Random Forest)
rf_pred = rf_model.predict(X_test)
cm_rf = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Predicted: Stay", "Predicted: Leave"],
    yticklabels=["Actual: Stay", "Actual: Leave"],
)
plt.title("Random Forest Confusion Matrix (Test Set N=294)")
plt.tight_layout()
plt.savefig("fig4_confusion_matrix.png", dpi=150)

print("\nAll 4 visualizations saved successfully!")
