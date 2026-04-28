import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load your dataset
df = pd.read_csv("duration_prediction_datasets/master.csv")

# Select features
feature_cols = [
    "avg_volume", 
    "peak_volume", 
    "volume_variance", 
    "zero_crossing_rate",
    "spectral_centroid", 
    "volume_delta", 
    "peak_delta", 
    "centroid_delta",
    "zcr_decay", 
    "rolling_avg_volume", 
    "rolling_peak_volume", 
    "rolling_decay",
    "centroid_volatility", 
    "is_chatter"
]

X = df[feature_cols]
y = df["duration_left_seconds"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")