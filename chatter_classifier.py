import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("tlc-apr-8-dataset.csv")

feature_cols = [
    "avg_volume",
    "peak_volume",
    "volume_variance",
    "zero_crossing_rate",
    "spectral_centroid",
    "rolling_avg_volume",
    "rolling_peak_volume"
]

CONFIDENCE_HIGH = 0.8
CONFIDENCE_LOW = 0.2
MAX_ITERATIONS = 50

# =========================
# NORMALIZATION
# =========================

scaler = StandardScaler()
normalized_features = pd.DataFrame(
    scaler.fit_transform(df[feature_cols]),
    columns=feature_cols
)

df = pd.concat([
    normalized_features, 
    df['is_chatter'].reset_indec(drop=True)],
    axis=1
)


# =========================
# ITERATIVE SELF-TRAINING LOOP
# =========================
for iteration in range(MAX_ITERATIONS):
    print(f"\n=== Iteration {iteration + 1} ===")

    labeled_df = df[df["is_chatter"].notna()].copy()
    unlabeled_df = df[df["is_chatter"].isna()].copy()

    print(f"Labeled rows: {len(labeled_df)}")
    print(f"Unlabeled rows: {len(unlabeled_df)}")

    # stop if fully labeled
    if len(unlabeled_df) == 0:
        print("All rows classified.")
        break

    # train model
    X_train = labeled_df[feature_cols]
    y_train = labeled_df["is_chatter"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # predict unlabeled rows
    X_unlabeled = unlabeled_df[feature_cols]
    probs = model.predict_proba(X_unlabeled)[:, 1]
    preds = model.predict(X_unlabeled)

    unlabeled_df["predicted_chatter"] = preds
    unlabeled_df["confidence"] = probs

    # keep only high-confidence rows
    confident_mask = (
        (unlabeled_df["confidence"] > CONFIDENCE_HIGH) |
        (unlabeled_df["confidence"] < CONFIDENCE_LOW)
    )

    confident_rows = unlabeled_df[confident_mask].copy()

    print(f"Confident new labels: {len(confident_rows)}")

    # stop if no confident rows found
    if len(confident_rows) == 0:
        print("No more high-confidence rows. Stopping.")
        break

    # assign pseudo labels back to original dataframe
    df.loc[confident_rows.index, "is_chatter"] = confident_rows["predicted_chatter"]

# =========================
# FINAL PASS: FORCE LABEL REMAINING ROWS
# =========================
remaining_unlabeled = df["is_chatter"].isna().sum()

if remaining_unlabeled > 0:
    print(f"\nFinal pass labeling remaining {remaining_unlabeled} rows.")

    labeled_df = df[df["is_chatter"].notna()].copy()
    unlabeled_df = df[df["is_chatter"].isna()].copy()

    X_train = labeled_df[feature_cols]
    y_train = labeled_df["is_chatter"]

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    X_unlabeled = unlabeled_df[feature_cols]
    final_preds = model.predict(X_unlabeled)

    df.loc[unlabeled_df.index, "is_chatter"] = final_preds

# =========================
# EXPORT COMPLETE CSV
# =========================
df.to_csv("fully_classified_chatter.csv", index=False)

print("\nSaved completed dataset to fully_classified_chatter.csv")