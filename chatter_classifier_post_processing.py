import pandas as pd
import numpy as np
import joblib

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "chatter_classifier.pkl"
SCALER_PATH = "scaler.pkl"
INPUT_CSV = "tlc-04-08-partial-data.csv"
OUTPUT_CSV = "tlc-04-08-classified-with-duration-left.csv"

FEATURE_COLS = [
    "avg_volume",
    "peak_volume",
    "volume_variance",
    "zero_crossing_rate",
    "spectral_centroid",
    "rolling_avg_volume",
    "rolling_peak_volume",
]

# -----------------------------
# LOAD MODEL AND DATASET
# -----------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

df = pd.read_csv(INPUT_CSV)

# Ensure time order
df = df.sort_values("timestamp_start").reset_index(drop=True)

# -----------------------------
# FILL is_chatter USING MODEL
# -----------------------------
X = scaler.transform(df[FEATURE_COLS])

df["is_chatter"] = model.predict(X).astype(int)

# Optional confidence/probability column
if hasattr(model, "predict_proba"):
    df["chatter_probability"] = model.predict_proba(X)[:, 1]

# -----------------------------
# COMPUTE chatter_streak
# -----------------------------
streaks = []
current_streak = 0

for val in df["is_chatter"]:
    if val == 1:
        current_streak += 1
    else:
        current_streak = 0
    streaks.append(current_streak)

df["chatter_streak"] = streaks
df["streak_log"] = np.log1p(df["chatter_streak"])

# -----------------------------
# COMPUTE duration_left
# -----------------------------
duration_left = []
count = 0

for val in reversed(df["is_chatter"]):
    if val == 1:
        count += 1
    else:
        count = 0
    duration_left.append(count)

df["duration_left"] = list(reversed(duration_left))

# If each row is 5 seconds, convert window count to seconds
df["duration_left_seconds"] = df["duration_left"] * 5

# -----------------------------
# SAVE OUTPUT
# -----------------------------
df.to_csv(OUTPUT_CSV, index=False)

print("Saved:", OUTPUT_CSV)
print("Rows:", len(df))
print("Chatter rows:", df["is_chatter"].sum())