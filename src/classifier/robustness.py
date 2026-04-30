import pandas as pd
import joblib
import numpy as np
from sklearn.inspection import permutation_importance

df_results = pd.read_csv("detailed_model_results.csv")

model = joblib.load('narrative_rf_model.joblib')

metadata_cols = ['label', 'source', 'theater', 'prediction', 'confidence', 'is_correct']
X = df_results.drop(columns=[c for c in metadata_cols if c in df_results.columns])
y = df_results['label']

print(f"✅ Imported {len(df_results)} rows and the trained Random Forest.")
print(f"Features available for analysis: {list(X.columns)}")
correlation_matrix = X[['morality_asymmetry', 'toxicity_score', 'ingroup_outgroup_ratio']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Independence Check: Are features unique?")
plt.show()

perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)

sorted_idx = perm_importance.importances_mean.argsort()
plt.figure(figsize=(10, 8))
plt.boxplot(perm_importance.importances[sorted_idx].T, vert=False, labels=X.columns[sorted_idx])
plt.title("Permutation Importance: Which feature actually breaks the model?")
plt.show()