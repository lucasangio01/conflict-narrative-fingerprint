import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import PartialDependenceDisplay
from src.utils.constants import ClassifierConfig
from src.utils.logging_config import get_logger

logger = get_logger("CLASSIFIER")


def main():
    # All analyses use the ablated (structural-only) model, which is the primary
    # result reported in Section 7.4. The full model is discussed in Section 7.6.

    model        = joblib.load(ClassifierConfig.ABLATED_MODEL_FILE)
    ablated_cols = joblib.load(ClassifierConfig.ABLATED_FEATURE_LIST_FILE)
    df_results   = pd.read_csv(ClassifierConfig.ABLATED_RESULTS_CSV, index_col=0)

    # Restrict to the ablated feature set (no toxicity_score)
    X_test = df_results[ablated_cols]
    y_test = df_results['label']

    logger.info(f"Primary model | Test set: {len(X_test)} rows | {X_test.shape[1]} features")
    logger.info(f"Features: {ablated_cols}")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax[0], color='darkred')
    ax[0].set_title("ROC Curve — Structural Core Model (no toxicity, test set)")
    ax[0].plot([0, 1], [0, 1], linestyle='--', color='grey', alpha=0.5)

    PrecisionRecallDisplay.from_estimator(model, X_test, y_test, ax=ax[1], color='darkblue')
    ax[1].set_title("Precision-Recall Curve — Structural Core Model (test set)")

    plt.tight_layout()
    plt.show()

    logger.info("Calculating SHAP values (this may take a moment)...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test, check_additivity=False)

    # Handle both older (list) and newer (3-D array) SHAP output formats
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]
    elif len(shap_values.shape) == 3:
        shap_to_plot = shap_values[:, :, 1]
    else:
        shap_to_plot = shap_values

    plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_to_plot, X_test, plot_type="dot", show=False)
    plt.title("SHAP: Feature Influence on 'Challenger' (label=1) Prediction\n"
              "[Structural core model — toxicity excluded]")
    plt.tight_layout()
    plt.show()

    # Bar chart version — cleaner for the thesis figure
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_to_plot, X_test, plot_type="bar", show=False)
    plt.title("SHAP: Mean Absolute Feature Importance\n"
              "[Structural core model — toxicity excluded]")
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots(figsize=(10, 6))
    PartialDependenceDisplay.from_estimator(model, X_test, ['morality_asymmetry'], ax=ax)
    ax.set_title("Partial Dependence: P(Challenger) vs Morality Asymmetry\n"
                 "[Structural core model]")
    ax.set_xlabel("Morality Asymmetry Score")
    ax.set_ylabel("Predicted Probability of Challenger Label")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    source_acc = (
        df_results
        .assign(is_correct=lambda d: d['label'] == d['prediction'])
        .groupby('source')['is_correct']
        .mean()
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=source_acc.values, y=source_acc.index,
                hue=source_acc.index, palette='viridis', legend=False)
    plt.title("Model Accuracy per Outlet (structural core model, test set)")
    plt.xlabel("Proportion Correctly Classified")
    plt.xlim(0, 1)
    plt.axvline(0.5, color='red', linestyle='--', alpha=0.5, label='Chance level')
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
