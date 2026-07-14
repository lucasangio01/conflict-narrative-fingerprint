import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, roc_auc_score
from sklearn.metrics import RocCurveDisplay
from src.utils.constants import ClassifierConfig


def main():
    df = pd.read_csv(ClassifierConfig.MERGED_DATA_CSV)
    print(f"Loaded {len(df)} rows | Label distribution:\n{df['label'].value_counts()}\n")

    X = df.drop(columns=['label', 'source', 'theater'])
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    test_indices = X_test.index.tolist()
    joblib.dump(test_indices, ClassifierConfig.TEST_INDICES_FILE)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # Two features are excluded from the primary (ablated) model:
    #
    #   toxicity_score  — measures surface aggression of the text as a whole,
    #                     pre-computed before any structural analysis; not
    #                     decomposed by entity or syntactic role.
    #
    #   chunk_sentiment — global RoBERTa sentiment score over the full chunk;
    #                     measures overall affective tone, not narrative
    #                     organization. Structurally identical in kind to
    #                     toxicity_score: both answer "how does this text feel?"
    #                     rather than "how is this narrative organized?"
    #
    # Both features fail the structural criterion: the CNF is defined in
    # Section 6.5.3 as a property of grammatical role assignment, relational
    # framing, and semantic organization, not of surface register intensity.
    # Research Question 1.3.5 asks specifically whether "structural and linguistic
    # features" are sufficient; global register signals are outside that scope.
    #
    # Both features are retained in the full model (robustness check, Section 7.6)
    # to quantify the register layer's contribution above the structural core.

    REGISTER_FEATURES = ClassifierConfig.REGISTER_FEATURES

    X_train_abl = X_train.drop(columns=REGISTER_FEATURES)
    X_test_abl  = X_test.drop(columns=REGISTER_FEATURES)
    joblib.dump(list(X_train_abl.columns), ClassifierConfig.ABLATED_FEATURE_LIST_FILE)

    print(f"\nPrimary model features ({len(X_train_abl.columns)}):")
    print(list(X_train_abl.columns))
    print(f"\nExcluded (register features): {REGISTER_FEATURES}")

    clf_params = ClassifierConfig.RF_PARAMS
    cv         = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # --- Primary model: ablated (structural features only) — reported in Section 7.4 as the answer to RQ 1.3.5 ---

    print("\n=== PRIMARY MODEL (structural features only, no register signals) ===")

    cv_abl = cross_validate(
        RandomForestClassifier(**clf_params),
        X_train_abl, y_train,
        cv=cv,
        scoring=['accuracy', 'f1_weighted', 'roc_auc'],
        return_train_score=False,
    )
    print("5-Fold CV Results (training set):")
    for metric, scores in cv_abl.items():
        if metric.startswith('test_'):
            print(f"  {metric[5:]:20s}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    clf_abl = RandomForestClassifier(**clf_params)
    clf_abl.fit(X_train_abl, y_train)
    joblib.dump(clf_abl, ClassifierConfig.ABLATED_MODEL_FILE)

    y_pred_abl = clf_abl.predict(X_test_abl)
    print("\nClassification Report — Primary Model (held-out test set):")
    print(classification_report(y_test, y_pred_abl))

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred_abl, ax=ax,
        display_labels=["Dominant (0)", "Challenger (1)"],
        colorbar=False,
    )
    ax.set_title("Confusion Matrix — Primary Model\n(no register features, test set)")
    plt.tight_layout()
    plt.show()

    df_test_abl               = df.loc[test_indices].copy()
    df_test_abl['prediction'] = clf_abl.predict(X_test_abl)
    df_test_abl['confidence'] = clf_abl.predict_proba(X_test_abl).max(axis=1)
    df_test_abl['is_correct'] = (df_test_abl['label'] == df_test_abl['prediction'])
    df_test_abl.to_csv(ClassifierConfig.ABLATED_RESULTS_CSV, index=True)
    print(f"Saved: {ClassifierConfig.ABLATED_RESULTS_CSV} (primary model, test set only)")

    # --- Robustness check: full model with all features including register signals — reported in Section 7.6 ---

    print("\n=== ROBUSTNESS CHECK (full model, all features) ===")

    cv_full = cross_validate(
        RandomForestClassifier(**clf_params),
        X_train, y_train,
        cv=cv,
        scoring=['accuracy', 'f1_weighted', 'roc_auc'],
        return_train_score=False,
    )
    print("5-Fold CV Results (training set):")
    for metric, scores in cv_full.items():
        if metric.startswith('test_'):
            print(f"  {metric[5:]:20s}: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")

    clf_full = RandomForestClassifier(**clf_params)
    clf_full.fit(X_train, y_train)
    joblib.dump(clf_full, ClassifierConfig.FULL_MODEL_FILE)

    y_pred_full = clf_full.predict(X_test)
    print("\nClassification Report — Full Model (held-out test set):")
    print(classification_report(y_test, y_pred_full))

    df_test_full               = df.loc[test_indices].copy()
    df_test_full['prediction'] = clf_full.predict(X_test)
    df_test_full['confidence'] = clf_full.predict_proba(X_test).max(axis=1)
    df_test_full['is_correct'] = (df_test_full['label'] == df_test_full['prediction'])
    df_test_full.to_csv(ClassifierConfig.FULL_RESULTS_CSV, index=True)
    print(f"Saved: {ClassifierConfig.FULL_RESULTS_CSV} (full model, test set only)")

    # --- Ablation comparison: ROC overlay + AUC decomposition for Section 7.6 ---

    auc_abl  = roc_auc_score(y_test, clf_abl.predict_proba(X_test_abl)[:, 1])
    auc_full = roc_auc_score(y_test, clf_full.predict_proba(X_test)[:, 1])

    print(f"\n--- Ablation Summary ---")
    print(f"  AUC structural core (no register):  {auc_abl:.3f}")
    print(f"  AUC full model (all features):      {auc_full:.3f}")
    print(f"  Register layer contribution:        {auc_full - auc_abl:+.3f}")
    print(f"  (register features: {REGISTER_FEATURES})")

    fig, ax = plt.subplots(figsize=(8, 6))
    RocCurveDisplay.from_estimator(
        clf_abl,  X_test_abl, y_test, ax=ax,
        name=f"Structural core (AUC = {auc_abl:.3f})",
        color='steelblue', linestyle='--',
    )
    RocCurveDisplay.from_estimator(
        clf_full, X_test, y_test, ax=ax,
        name=f"Full model with register features (AUC = {auc_full:.3f})",
        color='darkred',
    )
    ax.plot([0, 1], [0, 1], color='grey', linestyle=':', alpha=0.5)
    ax.set_title("Register Ablation: Structural Core vs Full Model")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(ClassifierConfig.ABLATION_ROC_PNG, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved: {ClassifierConfig.ABLATION_ROC_PNG}")

    print("\nAll models saved.")
    print(f"  Primary result   → {ClassifierConfig.ABLATED_RESULTS_CSV} + {ClassifierConfig.ABLATED_MODEL_FILE}")
    print(f"  Robustness check → {ClassifierConfig.FULL_RESULTS_CSV} + {ClassifierConfig.FULL_MODEL_FILE}")


if __name__ == "__main__":
    main()
