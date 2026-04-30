import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay
from sklearn.inspection import PartialDependenceDisplay

model = joblib.load('narrative_rf_model.joblib')
df_results = pd.read_csv("detailed_model_results.csv")

metadata_cols = ['label', 'source', 'theater', 'prediction', 'confidence', 'is_correct']
X = df_results.drop(columns=[c for c in metadata_cols if c in df_results.columns])
y = df_results['label']

fig, ax = plt.subplots(1, 2, figsize=(16, 6))
RocCurveDisplay.from_estimator(model, X, y, ax=ax[0], color='darkred')
ax[0].set_title("ROC Curve: Model Separability")

PrecisionRecallDisplay.from_estimator(model, X, y, ax=ax[1], color='darkblue')
ax[1].set_title("PR Curve: Challenger Identification Reliability")
plt.show()

print("🔎 Calculating SHAP values for narrative interpretation...")

explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X, check_additivity=False)

if isinstance(shap_values, list):
    shap_to_plot = shap_values[1]
elif len(shap_values.shape) == 3:
    shap_to_plot = shap_values[:, :, 1]
else:
    shap_to_plot = shap_values

plt.figure(figsize=(12, 10))
shap.summary_plot(shap_to_plot, X, plot_type="dot", show=False)
plt.title("SHAP: Feature Influence on 'Challenger' (1) Prediction")
plt.tight_layout()
plt.show()

print("📈 Plotting Partial Dependence for Morality...")
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(model, X, ['morality_asymmetry'], ax=ax)
ax.set_title("Tipping Point: Probability of 'Challenger' vs Morality Score")
plt.grid(True, alpha=0.3)
plt.show()

print("📊 Analyzing accuracy per outlet...")
df_results['is_correct'] = (df_results['label'] == df_results['prediction'])
source_acc = df_results.groupby('source')['is_correct'].mean().sort_values()

plt.figure(figsize=(10, 6))
sns.barplot(x=source_acc.values, y=source_acc.index, hue=source_acc.index, palette='viridis', legend=False)
plt.title("Model Reliability per News Outlet")
plt.xlabel("Accuracy (Proportion of Correct Predictions)")
plt.xlim(0, 1)
plt.show()