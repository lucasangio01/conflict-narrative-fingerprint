import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from src.utils.constants import ClassifierConfig


def main():
    # All analyses use the ablated (structural-only) model, which is the primary
    # result reported in Section 7.4.

    model        = joblib.load(ClassifierConfig.ABLATED_MODEL_FILE)
    ablated_cols = joblib.load(ClassifierConfig.ABLATED_FEATURE_LIST_FILE)
    df_results   = pd.read_csv(ClassifierConfig.ABLATED_RESULTS_CSV, index_col=0)

    X_test = df_results[ablated_cols]
    y_test = df_results['label']

    print(f"Primary model | Test set: {len(X_test)} rows")
    print(f"Features: {list(X_test.columns)}")

    # Covers all features in the ablated model to give a complete picture of
    # multicollinearity before interpreting permutation importance values.
    corr = X_test.corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap='coolwarm', center=0, linewidths=0.5,
        annot_kws={"size": 8}, square=True,
    )
    plt.title("Feature Correlation Matrix — Structural Core Model (test set)")
    plt.tight_layout()
    plt.show()

    # Flag pairs with |r| > 0.6
    high_corr = (
        corr.abs()
        .where(~mask)
        .stack()
        .reset_index()
        .rename(columns={"level_0": "Feature A", "level_1": "Feature B", 0: "|r|"})
        .query("`|r|` > 0.6")
        .sort_values("|r|", ascending=False)
    )
    if not high_corr.empty:
        print("\nHigh-correlation pairs (|r| > 0.6):")
        print(high_corr.to_string(index=False))
    else:
        print("\nNo feature pair exceeds |r| = 0.6.")

    print("\nCalculating permutation importance (10 repeats)...")
    perm_imp   = permutation_importance(
        model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
    )
    sorted_idx = perm_imp.importances_mean.argsort()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.boxplot(
        perm_imp.importances[sorted_idx].T,
        vert=False,
        labels=X_test.columns[sorted_idx],
        patch_artist=True,
        boxprops=dict(facecolor='steelblue', alpha=0.6),
        medianprops=dict(color='black'),
    )
    ax.axvline(0, color='red', linestyle='--', alpha=0.5, label='No effect baseline')
    ax.set_xlabel("Mean Accuracy Decrease")
    ax.set_title("Permutation Importance — Structural Core Model (test set, 10 repeats)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Summary table
    perm_summary = pd.DataFrame({
        "feature":   X_test.columns,
        "mean_drop": perm_imp.importances_mean,
        "std_drop":  perm_imp.importances_std,
    }).sort_values("mean_drop", ascending=False).reset_index(drop=True)

    print("\nPermutation Importance Summary:")
    print(perm_summary.to_string(index=False))


if __name__ == "__main__":
    main()
