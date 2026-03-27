"""
train_rf.py -- Model Training & Comparison Pipeline
====================================================
Trains Random Forest, Decision Tree, and XGBoost classifiers on IoT
network traffic data, performs cross-validation, generates evaluation
plots, and compares model performance.

Supports real-world datasets like ML-EdgeIIoT-dataset.csv with
automatic target column detection.

Usage
-----
    python train_rf.py --data data/ML-EdgeIIoT-dataset.csv --mode binary
    python train_rf.py --data data/ML-EdgeIIoT-dataset.csv --mode multiclass
"""

import argparse
import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from utils import (
    load_dataset,
    detect_target_column,
    prepare_features,
    engineer_features,
    encode_labels,
    scale_features,
    split_data,
    evaluate_model,
    compare_models,
    cross_validate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curve,
    plot_precision_recall_curve,
    compute_auc_score,
)


def train_and_evaluate(
    data_path: str,
    mode: str = "binary",
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    End-to-end training pipeline with 3-model comparison.

    Pipeline:
      1. Load & clean data
      2. Detect target column automatically
      3. Prepare numeric features (no leakage)
      4. Feature engineering (if applicable)
      5. Label encoding
      6. Train/test split (80-20 stratified) & scaling
      7. Train Random Forest + Decision Tree + XGBoost
      8. Cross-validation (5-fold on training data only)
      9. Evaluation on TEST SET ONLY
      10. Generate plots (confusion matrix, ROC, PR, feature importance)
      11. Model comparison table
      12. Save models & artifacts
    """
    # ----------------------------------------------------------------
    # 1. Load & preprocess
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  IoT ANOMALY DETECTION -- MODEL TRAINING PIPELINE")
    print("=" * 60)

    try:
        df = load_dataset(data_path)
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("   Make sure you placed your CSV in the data/ folder.")
        sys.exit(1)

    # ----------------------------------------------------------------
    # 2. Auto-detect target column
    # ----------------------------------------------------------------
    print(f"\n[STEP 2] Target Column Detection (mode={mode})")
    binary = mode == "binary"
    try:
        target_col = detect_target_column(df, mode=mode)
    except ValueError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    y_series = df[target_col]
    print(f"  Target: '{target_col}' | Unique values: {y_series.nunique()}")

    # ----------------------------------------------------------------
    # 3. Prepare features (safe numeric filtering)
    # ----------------------------------------------------------------
    print("\n[STEP 3] Feature Preparation")
    fdf = prepare_features(df, target_col)
    feature_names = list(fdf.columns)
    print(f"  {len(feature_names)} numeric features selected")

    # ----------------------------------------------------------------
    # 4. Feature engineering (only if source columns exist)
    # ----------------------------------------------------------------
    print("\n[STEP 4] Feature Engineering")
    fdf = engineer_features(fdf)
    feature_names = list(fdf.columns)
    new_feats = [c for c in feature_names if c in
                 ("byte_ratio", "src_dst_ratio", "packet_rate", "error_flag_interact")]
    if new_feats:
        print(f"  Added: {new_feats}")
    else:
        print("  No derived features added (source columns not present)")

    X = fdf.values.astype(np.float32)

    # ----------------------------------------------------------------
    # 5. Encode labels
    # ----------------------------------------------------------------
    print(f"\n[STEP 5] Label Encoding ({mode})")
    y, label_encoder, class_names = encode_labels(y_series, binary=binary)
    n_classes = len(class_names)

    # ----------------------------------------------------------------
    # 6. Train / test split & scaling
    # ----------------------------------------------------------------
    print("\n[STEP 6] Train/Test Split (80-20 Stratified) & Feature Scaling")
    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=test_size, random_state=random_state
    )

    scaler_path = f"models/scaler_{mode}.pkl"
    X_train_s, X_test_s, scaler = scale_features(
        X_train, X_test, save_path=scaler_path
    )

    # ----------------------------------------------------------------
    # 7. Train models
    # ----------------------------------------------------------------
    os.makedirs("models", exist_ok=True)

    # --- Random Forest ---
    print("\n[STEP 7a] Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    rf.fit(X_train_s, y_train)
    print("  [OK] Random Forest trained.")

    # --- Decision Tree ---
    print("[STEP 7b] Training Decision Tree...")
    dt = DecisionTreeClassifier(
        random_state=random_state,
        class_weight="balanced",
        max_depth=15,
    )
    dt.fit(X_train_s, y_train)
    print("  [OK] Decision Tree trained.")

    # --- XGBoost ---
    print("[STEP 7c] Training XGBoost...")
    xgb_params = dict(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        learning_rate=0.1,
        max_depth=6,
        eval_metric="logloss" if binary else "mlogloss",
    )
    if binary:
        xgb_params["objective"] = "binary:logistic"
        xgb_params["scale_pos_weight"] = float(
            np.sum(y_train == 0) / max(np.sum(y_train == 1), 1)
        )
    else:
        xgb_params["objective"] = "multi:softprob"
        xgb_params["num_class"] = n_classes

    xgb = XGBClassifier(**xgb_params)
    xgb.fit(X_train_s, y_train)
    print("  [OK] XGBoost trained.")

    # ----------------------------------------------------------------
    # 8. Cross-validation (on TRAINING data only -- no leakage)
    # ----------------------------------------------------------------
    print("\n[STEP 8] Cross-Validation (5-fold, training data only)")

    models_for_cv = {
        "Random Forest": RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            n_jobs=-1, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state, class_weight="balanced", max_depth=15),
        "XGBoost": XGBClassifier(**xgb_params),
    }

    cv_results = {}
    for name, model_cv in models_for_cv.items():
        print(f"\n  -- {name} --")
        cv_results[name] = cross_validate_model(model_cv, X_train_s, y_train, cv=5)

    # ----------------------------------------------------------------
    # 9. Evaluate on TEST SET ONLY (no training data evaluation)
    # ----------------------------------------------------------------
    print("\n[STEP 9] Test Set Evaluation (test data only -- no leakage)")

    trained_models = {"Random Forest": rf, "Decision Tree": dt, "XGBoost": xgb}
    all_metrics = {}
    all_preds = {}
    all_probas = {}

    for name, model in trained_models.items():
        print(f"\n  -- {name} --")
        y_pred = model.predict(X_test_s)
        y_proba = model.predict_proba(X_test_s)
        metrics = evaluate_model(y_test, y_pred, class_names=class_names)

        auc_val = compute_auc_score(y_test, y_proba, binary=binary)
        metrics["auc"] = round(auc_val, 4)
        print(f"  AUC: {auc_val:.4f}")

        all_metrics[name] = metrics
        all_preds[name] = y_pred
        all_probas[name] = y_proba

    # ----------------------------------------------------------------
    # 10. Plots -- Confusion Matrix, ROC, PR curve, Feature Importance
    #     All plots use TEST data only
    # ----------------------------------------------------------------
    print("\n[STEP 10] Generating Evaluation Plots (test data)")

    model_tags = {"Random Forest": "rf", "Decision Tree": "dt", "XGBoost": "xgb"}
    for name, tag in model_tags.items():
        plot_confusion_matrix(
            y_test, all_preds[name], class_names=class_names,
            save_path=f"models/confusion_matrix_{tag}_{mode}.png",
        )
        plot_roc_curve(
            y_test, all_probas[name], class_names=class_names, binary=binary,
            save_path=f"models/roc_curve_{tag}_{mode}.png",
        )
        plot_precision_recall_curve(
            y_test, all_probas[name], class_names=class_names, binary=binary,
            save_path=f"models/pr_curve_{tag}_{mode}.png",
        )
        if hasattr(trained_models[name], "feature_importances_"):
            plot_feature_importance(
                trained_models[name].feature_importances_,
                feature_names,
                top_n=min(20, len(feature_names)),
                save_path=f"models/feature_importance_{tag}_{mode}.png",
            )

    # Permutation importance (Random Forest, on test data)
    plot_permutation_importance(
        rf, X_test_s, y_test,
        feature_names,
        top_n=min(15, len(feature_names)),
        save_path=f"models/permutation_importance_rf_{mode}.png",
    )

    # ----------------------------------------------------------------
    # 11. Model comparison
    # ----------------------------------------------------------------
    print("\n[STEP 11] Model Comparison")
    comparison_df = compare_models(all_metrics)

    # ----------------------------------------------------------------
    # 12. Save models & artifacts
    # ----------------------------------------------------------------
    print("\n[STEP 12] Saving Models & Artifacts")

    joblib.dump(rf, "models/random_forest.pkl")
    print("  [OK] Random Forest -> models/random_forest.pkl")

    joblib.dump(dt, "models/decision_tree.pkl")
    print("  [OK] Decision Tree -> models/decision_tree.pkl")

    joblib.dump(xgb, "models/xgboost.pkl")
    print("  [OK] XGBoost -> models/xgboost.pkl")

    if label_encoder is not None:
        le_path = f"models/label_encoder_{mode}.pkl"
        joblib.dump(label_encoder, le_path)
        print(f"  [OK] Label encoder -> {le_path}")

    # Save feature names for Streamlit compatibility
    joblib.dump(feature_names, f"models/feature_names_{mode}.pkl")
    print(f"  [OK] Feature names -> models/feature_names_{mode}.pkl")

    # Save comparison table
    comparison_df.to_csv(f"models/model_comparison_{mode}.csv")
    print(f"  [OK] Comparison table -> models/model_comparison_{mode}.csv")

    # Save cross-validation results
    cv_summary = {}
    for name, cv_res in cv_results.items():
        cv_summary[name] = {k: f"{v['mean']:.4f} +/- {v['std']:.4f}"
                            for k, v in cv_res.items()}
    cv_df = pd.DataFrame(cv_summary).T
    cv_df.to_csv(f"models/cross_validation_{mode}.csv")
    print(f"  [OK] CV results -> models/cross_validation_{mode}.csv")

    print("\n" + "=" * 60)
    print("  TRAINING PIPELINE COMPLETE!")
    print("=" * 60 + "\n")

    return trained_models, all_metrics


# -----------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ML models for IoT Anomaly Detection"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ML-EdgeIIoT-dataset.csv",
        help="Path to CSV dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["binary", "multiclass"],
        default="binary",
        help="Classification mode: binary or multiclass",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=100,
        help="Number of trees (default: 100)",
    )
    args = parser.parse_args()

    train_and_evaluate(
        data_path=args.data,
        mode=args.mode,
        n_estimators=args.estimators,
    )
