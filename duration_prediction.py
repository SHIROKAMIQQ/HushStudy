import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
  mean_squared_error, 
  mean_absolute_error
)

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
]

X = df[feature_cols]
y = df["duration_left_seconds"]

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

# Hyperparameter tuning
pipeline = Pipeline([
  ("model", RandomForestRegressor(random_state=42))
])

param_grid = {
  "model__n_estimators": [50, 100, 200],
  "model__max_depth": [None, 10, 20],
  "model__min_samples_split": [2, 5],
  "model__min_samples_leaf": [1, 2]
}

grid_search = GridSearchCV(
  estimator=pipeline,
  param_grid=param_grid,
  cv=5,
  scoring="neg_mean_absolute_error",
  verbose=1,
  n_jobs=-1
)
grid_search.fit(X_train, y_train)
model = grid_search.best_estimator_

print("BEST PARAMETERS FOR DURATION PREDICTION RandomForestRegressor:")
print(grid_search.best_params_)

# Evaluation
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nMETRICS FOR CHATTER CLASSIFIER LogisticRegression:")
print(f"MSE: {mse}")
print(f"MAE: {mae}")