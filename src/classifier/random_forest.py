import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("classification_data.csv")

X = df.drop(columns=['label', 'source', 'theater'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("📊 Classification Report (Hold-out Test Set):")
print(classification_report(y_test, y_pred))

joblib.dump(clf, 'narrative_rf_model.joblib')

df['prediction'] = clf.predict(X)
df['confidence'] = clf.predict_proba(X).max(axis=1)

df.to_csv("detailed_model_results.csv", index=False)

print("\n🧠 Model and 'detailed_model_results.csv' are ready for Step 2.")