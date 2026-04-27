import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# =========================
# LOAD DATASET
# =========================
DATASET_CSV="datasets/master.csv"
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
# NORMALIZATION
# =========================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =========================
# TRAIN MODEL
# =========================
model = LogisticRegression(
  max_iter=1000,
  class_weight="balanced"
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# EVALUATION
# =========================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)