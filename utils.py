"""
utils.py — Utility Functions for IoT Network Anomaly Detection System
=====================================================================
Contains reusable helpers for:
  - Data loading & preprocessing
  - Feature engineering (derived features)
  - Target column auto-detection (Attack_label / Attack_type / label)
  - Label encoding & feature scaling
  - Train-test splitting
  - Cross-validation (stratified k-fold)
  - Model evaluation (metrics, confusion matrix, ROC, PR curves)
  - Visualisation (feature importance, permutation importance)
  - Model comparison
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # Non-interactive backend (safe for Streamlit)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# 1. DATA LOADING & PREPROCESSING
# ---------------------------------------------------------------------------

def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load a CSV dataset and perform basic cleaning.

    Steps:
      1. Read CSV into a DataFrame (with low_memory=False for mixed types).
      2. Drop duplicate rows.
      3. Replace infinities with NaN.
      4. Fill missing numeric values with column medians.
      5. Fill missing categorical values with column modes.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath, low_memory=False)
    print(f"[INFO] Loaded dataset: {filepath}")
    print(f"       Shape        : {df.shape}")
    print(f"       Duplicates   : {df.duplicated().sum()}")

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Replace infinities with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Handle missing values -- numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Handle missing values -- categorical columns
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    print(f"[INFO] After cleaning: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# 2. TARGET COLUMN AUTO-DETECTION
# ---------------------------------------------------------------------------

def detect_target_column(df: pd.DataFrame, mode: str = "binary") -> str:
    """
    Auto-detect the target column based on classification mode.

    Priority:
      - binary:     Attack_label > label
      - multiclass: Attack_type  > label

    Raises ValueError if no suitable target column found.
    """
    if mode == "binary":
        candidates = ["Attack_label", "attack_label", "label", "Label"]
    else:
        candidates = ["Attack_type", "attack_type", "label", "Label"]

    for col in candidates:
        if col in df.columns:
            print(f"[INFO] Target column detected: '{col}' (mode={mode})")
            return col

    raise ValueError(
        f"No target column found for mode='{mode}'. "
        f"Expected one of: {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


# ---------------------------------------------------------------------------
# 3. FEATURE PREPARATION (safe numeric filtering)
# ---------------------------------------------------------------------------

def prepare_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Prepare feature matrix by:
      1. Dropping ALL target-related columns (Attack_label, Attack_type, label)
      2. Keeping ONLY numeric columns
      3. Replacing inf/NaN with 0
      4. Asserting no target leakage
    """
    # Columns to always drop (target + related)
    drop_cols = {"Attack_label", "Attack_type", "attack_label", "attack_type",
                 "label", "Label"}
    cols_to_drop = [c for c in df.columns if c in drop_cols]

    fdf = df.drop(columns=cols_to_drop, errors="ignore")

    # Keep only numeric features
    fdf = fdf.select_dtypes(include=["number"])

    # Clean any remaining inf/nan
    fdf.replace([np.inf, -np.inf], np.nan, inplace=True)
    fdf.fillna(0, inplace=True)

    # Leakage guard
    for col in drop_cols:
        assert col not in fdf.columns, f"DATA LEAKAGE: '{col}' still in features!"

    print(f"[INFO] Feature matrix: {fdf.shape[1]} numeric features")
    return fdf


# ---------------------------------------------------------------------------
# 4. FEATURE ENGINEERING
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that capture traffic behaviour patterns.
    Only adds features if source columns exist.
    """
    df = df.copy()

    if "src_bytes" in df.columns and "dst_bytes" in df.columns:
        df["byte_ratio"] = (df["dst_bytes"] / (df["src_bytes"] + 1)).round(4)
        df["src_dst_ratio"] = (df["src_bytes"] / (df["dst_bytes"] + 1)).round(4)

    if "packet_size" in df.columns and "duration" in df.columns:
        df["packet_rate"] = (df["packet_size"] / (df["duration"] + 0.01)).round(4)

    if "error_rate" in df.columns and "wrong_fragment" in df.columns:
        df["error_flag_interact"] = (df["error_rate"] * df["wrong_fragment"]).round(4)

    return df


# ---------------------------------------------------------------------------
# 5. LABEL ENCODING
# ---------------------------------------------------------------------------

def encode_labels(y_series: pd.Series, binary: bool = True):
    """
    Encode the target series.

    Parameters
    ----------
    y_series : The target column as a pandas Series.
    binary   : If True  -> 0 = Normal, 1 = Attack (or use existing 0/1).
               If False -> LabelEncoder for multiclass.

    Returns
    -------
    y           : np.ndarray   -- encoded labels
    label_encoder : LabelEncoder or None
    class_names : list[str]
    """
    if binary:
        # If already numeric 0/1
        if set(y_series.unique()) <= {0, 1}:
            y = y_series.values.astype(np.int64)
        else:
            normal_keywords = {"normal", "benign", "Normal", "BENIGN", "NORMAL"}
            y = np.array([0 if str(v).strip() in normal_keywords else 1
                          for v in y_series])
        class_names = ["Normal", "Attack"]
        label_encoder = None
    else:
        le = LabelEncoder()
        y = le.fit_transform(y_series.astype(str))
        class_names = list(le.classes_)
        label_encoder = le

    print(f"[INFO] Label encoding ({'binary' if binary else 'multiclass'})")
    print(f"       Classes ({len(class_names)}): {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
    return y, label_encoder, class_names


# ---------------------------------------------------------------------------
# 6. FEATURE SCALING
# ---------------------------------------------------------------------------

def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    save_path: str = "models/scaler.pkl",
):
    """
    Standardise features using StandardScaler fitted on training data.
    Saves the fitted scaler for later use in the web app.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(scaler, save_path)
    print(f"[INFO] Scaler saved to {save_path}")
    return X_train_scaled, X_test_scaled, scaler


# ---------------------------------------------------------------------------
# 7. TRAIN / TEST SPLIT HELPER
# ---------------------------------------------------------------------------

def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Stratified train-test split with sensible defaults."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# 8. CROSS-VALIDATION
# ---------------------------------------------------------------------------

def cross_validate_model(model, X, y, cv: int = 5):
    """
    Run stratified k-fold cross-validation and return mean +/- std metrics.

    Returns dict with keys: accuracy, precision, recall, f1
    Each value is a dict with 'mean', 'std', and 'scores'.
    """
    print(f"\n[INFO] Running {cv}-fold stratified cross-validation...")

    scoring = {
        "accuracy":  "accuracy",
        "precision": "precision_weighted",
        "recall":    "recall_weighted",
        "f1":        "f1_weighted",
    }

    cv_results = cross_validate(
        model, X, y,
        cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42),
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    results = {}
    print(f"\n  {'Metric':12s} {'Mean':>8s} {'Std':>8s}")
    print(f"  {'─' * 30}")

    metric_display = {
        "accuracy":  "Accuracy",
        "precision": "Precision",
        "recall":    "Recall",
        "f1":        "F1 Score",
    }

    for key, display in metric_display.items():
        scores = cv_results[f"test_{key}"]
        results[key] = {
            "mean": float(np.mean(scores)),
            "std":  float(np.std(scores)),
            "scores": scores.tolist(),
        }
        print(f"  {display:12s} {np.mean(scores):8.4f} +/- {np.std(scores):.4f}")

    return results


# ---------------------------------------------------------------------------
# 9. EVALUATION HELPERS
# ---------------------------------------------------------------------------

def evaluate_model(y_true, y_pred, class_names=None):
    """
    Compute and print standard classification metrics.

    Returns dict with accuracy, precision, recall, f1.
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("            EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("=" * 50)

    # Derive labels from actual data to avoid mismatch
    unique_labels = sorted(set(y_true) | set(y_pred))
    if class_names is not None and len(class_names) >= len(unique_labels):
        names = [class_names[i] if i < len(class_names) else str(i)
                 for i in unique_labels]
    else:
        names = [str(i) for i in unique_labels]

    print("\nClassification Report:\n")
    print(classification_report(
        y_true, y_pred, labels=unique_labels,
        target_names=names, zero_division=0
    ))

    return {
        "accuracy":  round(acc, 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "f1_score":  round(f1, 4),
    }


def compare_models(results_dict: dict) -> pd.DataFrame:
    """
    Compare multiple models given a dict of {model_name: metrics_dict}.
    Returns a formatted DataFrame for display.
    """
    rows = []
    for name, metrics in results_dict.items():
        rows.append({
            "Model": name,
            "Accuracy":  metrics.get("accuracy", 0),
            "Precision": metrics.get("precision", 0),
            "Recall":    metrics.get("recall", 0),
            "F1 Score":  metrics.get("f1_score", 0),
            "AUC":       metrics.get("auc", 0),
        })

    df = pd.DataFrame(rows).set_index("Model")

    print("\n" + "=" * 70)
    print("              MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    return df


# ---------------------------------------------------------------------------
# 10. CONFUSION MATRIX
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names=None,
    save_path: Optional[str] = None,
):
    """Plot (and optionally save) a heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Confusion matrix saved to {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 11. ROC CURVE & AUC
# ---------------------------------------------------------------------------

def plot_roc_curve(
    y_true,
    y_score,
    class_names=None,
    binary: bool = True,
    save_path: Optional[str] = None,
):
    """
    Plot ROC curve for binary or multiclass (one-vs-rest).
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    if binary:
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color="steelblue", lw=2,
                label=f"ROC curve (AUC = {roc_auc:.4f})")
    else:
        n_classes = y_score.shape[1] if y_score.ndim == 2 else len(set(y_true))
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc_i = auc(fpr, tpr)
            label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            ax.plot(fpr, tpr, color=colors[i], lw=2,
                    label=f"{label} (AUC = {roc_auc_i:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] ROC curve saved to {save_path}")

    plt.close(fig)
    return fig


def compute_auc_score(y_true, y_score, binary: bool = True) -> float:
    """Compute AUC score for binary or multiclass."""
    try:
        if binary:
            if y_score.ndim == 2:
                y_score = y_score[:, 1]
            return float(roc_auc_score(y_true, y_score))
        else:
            return float(roc_auc_score(
                y_true, y_score, multi_class="ovr", average="weighted"
            ))
    except ValueError:
        return 0.0


# ---------------------------------------------------------------------------
# 12. PRECISION-RECALL CURVE
# ---------------------------------------------------------------------------

def plot_precision_recall_curve(
    y_true,
    y_score,
    class_names=None,
    binary: bool = True,
    save_path: Optional[str] = None,
):
    """Plot Precision-Recall curve for binary or multiclass."""
    fig, ax = plt.subplots(figsize=(8, 6))

    if binary:
        if y_score.ndim == 2:
            y_score = y_score[:, 1]
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        ax.plot(rec, prec, color="steelblue", lw=2,
                label=f"PR curve (AP = {ap:.4f})")
    else:
        n_classes = y_score.shape[1] if y_score.ndim == 2 else len(set(y_true))
        y_bin = label_binarize(y_true, classes=list(range(n_classes)))
        colors = plt.cm.Set2(np.linspace(0, 1, n_classes))

        for i in range(n_classes):
            prec, rec, _ = precision_recall_curve(y_bin[:, i], y_score[:, i])
            ap = average_precision_score(y_bin[:, i], y_score[:, i])
            label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
            ax.plot(rec, prec, color=colors[i], lw=2,
                    label=f"{label} (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Precision-Recall curve saved to {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 13. FEATURE IMPORTANCE
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importances: np.ndarray,
    feature_names,
    top_n: int = 20,
    save_path: Optional[str] = None,
):
    """Bar chart of the top-N most important features."""
    indices = np.argsort(importances)[::-1][:top_n]

    names_arr = np.array(feature_names)
    top_names = names_arr[indices].tolist()
    top_vals  = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_vals[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Feature importance plot saved to {save_path}")

    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# 14. PERMUTATION IMPORTANCE
# ---------------------------------------------------------------------------

def plot_permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names,
    top_n: int = 15,
    n_repeats: int = 10,
    save_path: Optional[str] = None,
):
    """
    Compute and plot permutation importance -- measures how much
    each feature degrades performance when shuffled.
    """
    print("[INFO] Computing permutation importance (this may take a moment)...")

    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        random_state=42,
        n_jobs=-1,
        scoring="accuracy",
    )

    sorted_idx = result.importances_mean.argsort()[::-1][:top_n]
    names_arr = np.array(feature_names)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=names_arr[sorted_idx],
    )
    ax.set_xlabel("Decrease in Accuracy", fontsize=12)
    ax.set_title("Permutation Importance", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"[INFO] Permutation importance plot saved to {save_path}")

    plt.close(fig)
    return fig, result
