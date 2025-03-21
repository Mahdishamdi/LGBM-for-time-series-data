import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

# Load and prepare dataset
df = pd.read_csv("./data/Demo.csv")
df["timestamp"] = pd.to_datetime(df["Time (UTC)"])
df = df.sort_values("timestamp")

# Feature engineering
df["hour"] = df["timestamp"].dt.hour
df["minute"] = df["timestamp"].dt.minute
df["dayofweek"] = df["timestamp"].dt.dayofweek

# Lag features
df["demand_lag1"] = df["meter/site_demand"].shift(1)
df["demand_lag2"] = df["meter/site_demand"].shift(2)
df["pv_lag1"] = df["meter/pv_power"].shift(1)

# Rolling features
df["demand_rolling_mean_3"] = df["meter/site_demand"].rolling(window=3).mean()
df["demand_rolling_std_3"] = df["meter/site_demand"].rolling(window=3).std()

# Drop NA after creating lag/rolling features
df.dropna(inplace=True)

# Define target and features
target = "meter/site_demand"
features = [
    "hour", "minute", "dayofweek",
    "meter/pv_power", "demand_lag1", "demand_lag2", "pv_lag1",
    "demand_rolling_mean_3", "demand_rolling_std_3"
]

X = df[features]
y = df[target]

# Train-test split without shuffling to preserve time order
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Prepare LightGBM datasets
train_data = lgb.Dataset(X_train, label=y_train)
valid_data = lgb.Dataset(X_test, label=y_test)

# Train the model with improved parameters
params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbosity": -1
}
model = lgb.train(params, train_data, valid_sets=[valid_data], num_boost_round=300, early_stopping_rounds=20)

# Predict and evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

# Plot results
plt.figure(figsize=(12, 5))
plt.plot(y_test.values[:100], label="Actual")
plt.plot(y_pred[:100], label="Predicted")
plt.title(f"Improved Site Demand Forecasting (MAE: {mae:.2f})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
