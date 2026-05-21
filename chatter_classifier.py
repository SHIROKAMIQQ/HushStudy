import os

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
  accuracy_score, 
  precision_score, 
  recall_score,
  confusion_matrix,
  ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# =========================
# LOAD DATASET
# =========================
GRAPH_DIR = "model_graphs"
os.makedirs(GRAPH_DIR, exist_ok=True)

DATASET_CSV="chatter_classifier_datasets/master.csv"
df = pd.read_csv(DATASET_CSV)

feature_cols = [
    "avg_volume",
    "peak_volume",
    "volume_variance",
    "zero_crossing_rate",
    "spectral_centroid",
    "rolling_avg_volume",
    "rolling_peak_volume"
]

X = df[feature_cols]
y = df["is_chatter"]

# =========================
# TRAIN-TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
  X, y,
  test_size=0.2,
  random_state=42,
  stratify=y
)

# =========================
# HYPERPARAMETER TUNING
# =========================

pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("model", LogisticRegression(max_iter=1000))
])

param_grid = {
  "model__C": [0.01, 0.1, 1, 10, 100],
  "model__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
  estimator=pipeline,
  param_grid=param_grid,
  cv=5,
  scoring="f1",
  verbose=1,
  n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("BEST PARAMETERS FOR CHATTER CLASSIFIER LogisticRegression:")
print(grid_search.best_params_)

model = grid_search.best_estimator_

# =========================
# EVALUATION
# =========================
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nMETRICS FOR CHATTER CLASSIFIER LogisticRegression:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Chatter_Classifier Confusion Matrix")
conf_matrix_path = f"{GRAPH_DIR}/ChatterClassifier_ConfusionMatrix.png"
plt.savefig(conf_matrix_path)
plt.close()
print(f"Saved confusion matrix to: {conf_matrix_path}")

print("\n")