import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("tlc-apr-8-dataset2_streaked.csv")

# ensure correct ordering (IMPORTANT for time features)
df = df.sort_values("timestamp_start").reset_index(drop=True)

# -----------------------------
# CREATE TARGET LABEL
# -----------------------------
df["continues_next"] = df["is_chatter"].shift(-1).fillna(0).astype(int)

# -----------------------------
# FEATURE SET (MUST MATCH EXTRACTOR)
# -----------------------------
FEATURE_COLS = [
    "avg_volume",
    "peak_volume",
    "volume_variance",
    "zero_crossing_rate",
    "spectral_centroid",
    "rolling_avg_volume",
    "rolling_peak_volume",
    "chatter_streak",
    "streak_log",

    # NEW FEATURES
    "volume_delta",
    "peak_delta",
    "centroid_delta",
    "zcr_decay",
    "rolling_decay",
    "near_silence",
    "centroid_volatility"
]

# -----------------------------
# FILTER VALID TRAINING ROWS
# -----------------------------
df_cont = df[df["is_chatter"].notna()].copy()

X = df_cont[FEATURE_COLS]
y = df_cont["continues_next"]

# -----------------------------
# TIME-BASED SPLIT (NO LEAKAGE)
# -----------------------------
split_idx = int(len(X) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

# -----------------------------
# MODEL
# -----------------------------
continuation_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

continuation_model.fit(X_train, y_train)

# -----------------------------
# EVALUATION
# -----------------------------
y_pred = continuation_model.predict(X_test)

print("\nContinuation Model Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# FEATURE IMPORTANCE (DEBUG)
# -----------------------------
importance = pd.DataFrame({
    "feature": FEATURE_COLS,
    "importance": continuation_model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance)

# -----------------------------
# DURATION PREDICTION
# -----------------------------
def predict_remaining_duration(feature_row, max_seconds=60, step_duration=1.0):
    """
    Estimates remaining chatter duration using survival-style simulation.
    """

    row = feature_row.copy()
    expected_duration = 0.0
    survival_prob = 1.0

    steps = int(max_seconds / step_duration)

    for _ in range(steps):

        # keep streak-consistency
        row["streak_log"] = np.log1p(row["chatter_streak"])

        p_continue = continuation_model.predict_proba(row)[0, 1]

        survival_prob *= p_continue
        expected_duration += survival_prob * step_duration

        # simulate next time step
        row["chatter_streak"] += 1

    return expected_duration

# -----------------------------
# SAMPLE INPUT
# -----------------------------
sample = pd.DataFrame([{
    "avg_volume": 0.3,
    "peak_volume": 0.8,
    "volume_variance": 0.1,
    "zero_crossing_rate": 0.05,
    "spectral_centroid": 2200,
    "rolling_avg_volume": 0.28,
    "rolling_peak_volume": 0.75,
    "chatter_streak": 8,

    # derived features
    "streak_log": np.log1p(8),
    "volume_delta": 0.0,
    "peak_delta": 0.0,
    "centroid_delta": 0.0,
    "zcr_decay": 0.0,
    "rolling_decay": 0.0,
    "near_silence": 0,
    "centroid_volatility": 0.0
}])

predicted = predict_remaining_duration(sample)

print(f"\nPredicted remaining duration: {predicted:.2f} seconds")

import joblib

joblib.dump(continuation_model, "continuation_model.pkl")
print("Model saved as continuation_model.pkl")