import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import yfinance as yf

# ========================
# FIXED METRICS FUNCTION
# ========================
def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
    r2 = r2_score(y_true, y_pred)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2}

# ========================
# Load Tesla Stock Data
# ========================
symbol = "TSLA"
start_date = "2020-01-01"
end_date = datetime.today().strftime('%Y-%m-%d')

tesla_data = yf.download(symbol, start=start_date, end=end_date)
tesla_data = tesla_data.copy()

print("Tesla Data Loaded Successfully!")
print(tesla_data.head())

# Ensure DateTimeIndex
if not isinstance(tesla_data.index, pd.DatetimeIndex):
    tesla_data.index = pd.to_datetime(tesla_data.index)

tesla_data = tesla_data.sort_index()

# Handle Missing Values 
tesla_data = tesla_data.ffill().bfill()

# ========================
# Outlier Handling (Returns)
# ========================
tesla_data["Return"] = tesla_data["Close"].pct_change()
ret = tesla_data["Return"]
z = (ret - ret.mean()) / ret.std(ddof=0)
tesla_data["Return_clipped"] = ret.mask(z > 3, ret.quantile(0.997)).mask(z < -3, ret.quantile(0.003))
tesla_data["Return_clipped"] = tesla_data["Return_clipped"].fillna(0.0)

# ========================
# Visualization
# ========================
plt.figure(figsize=(12, 4))
plt.plot(tesla_data.index, tesla_data["Close"], label="Close")
plt.title("TSLA Close Price")
plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.show()

# Moving Averages
tesla_data["SMA_20"] = tesla_data["Close"].rolling(20).mean()
tesla_data["SMA_50"] = tesla_data["Close"].rolling(50).mean()

plt.figure(figsize=(12, 4))
plt.plot(tesla_data.index, tesla_data["Close"], label="Close")
plt.plot(tesla_data.index, tesla_data["SMA_20"], label="SMA 20")
plt.plot(tesla_data.index, tesla_data["SMA_50"], label="SMA 50")
plt.title("TSLA Close with Moving Averages")
plt.xlabel("Date"); plt.ylabel("Price"); plt.legend(); plt.show()

# ========================
# Feature Engineering
# ========================
fe = tesla_data.copy()

def roll_feat(series, window, func="mean"):
    if func == "mean":
        return series.rolling(window).mean().shift(1)
    if func == "std":
        return series.rolling(window).std().shift(1)
    if func == "min":
        return series.rolling(window).min().shift(1)
    if func == "max":
        return series.rolling(window).max().shift(1)
    raise ValueError("Unknown func")

# Lags of Close (1..10 days for better context)
for lag in range(1, 11):
    fe[f"Close_lag_{lag}"] = fe["Close"].shift(lag)

# Lags of returns (1..5 days)
for lag in range(1, 6):
    fe[f"Ret_lag_{lag}"] = fe["Return_clipped"].shift(lag)

# Rolling stats on Close
fe["roll_mean_5"]  = roll_feat(fe["Close"], 5, "mean")
fe["roll_mean_10"] = roll_feat(fe["Close"], 10, "mean")
fe["roll_std_5"]   = roll_feat(fe["Close"], 5, "std")
fe["roll_std_10"]  = roll_feat(fe["Close"], 10, "std")
fe["roll_min_14"]  = roll_feat(fe["Close"], 14, "min")
fe["roll_max_14"]  = roll_feat(fe["Close"], 14, "max")

# Volume features
fe["Vol_lag_1"]   = fe["Volume"].shift(1)
fe["Vol_sma_5"]   = roll_feat(fe["Volume"], 5, "mean")

# Calendar features
fe["dayofweek"] = fe.index.dayofweek
fe["month"]     = fe.index.month

# Target: predict next-day Close
fe["y_next"] = fe["Close"].shift(-1)

# Drop NaN rows
fe = fe.dropna().copy()

# Final features
feature_cols = [col for col in fe.columns if col not in ["y_next", "Close", "Return", "Return_clipped", "SMA_20", "SMA_50"]]
X = fe[feature_cols].astype(float)
y = fe["y_next"].astype(float)

# ========================
# Train-Validation-Test Split
# ========================
N = len(fe)
train_end = int(N * 0.70)
val_end   = int(N * 0.85)

X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
X_val,   y_val   = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
X_test,  y_test  = X.iloc[val_end:], y.iloc[val_end:]

print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# ========================
# Ridge Regression with Feature Selection + Hyperparameter Tuning
# ========================
ridge_pipeline = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("feature_selection", SelectKBest(score_func=f_regression, k=25)),  # Select top 25 features
    ("ridge", Ridge())
])

# Hyperparameter tuning using TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "ridge__alpha": [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
    "feature_selection__k": [15, 20, 25, 30]
}

grid_search = GridSearchCV(ridge_pipeline, param_grid, cv=tscv, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\nBest Ridge Parameters:", grid_search.best_params_)
best_ridge_model = grid_search.best_estimator_

# ========================
# Train Linear Regression for baseline
# ========================
lr_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
lr_pipe.fit(X_train, y_train)

# ========================
# Predictions
# ========================
yhat_val_lr  = lr_pipe.predict(X_val)
yhat_test_lr = lr_pipe.predict(X_test)

yhat_val_ridge  = best_ridge_model.predict(X_val)
yhat_test_ridge = best_ridge_model.predict(X_test)

# Metrics
val_metrics_lr = metrics(y_val, yhat_val_lr)
test_metrics_lr = metrics(y_test, yhat_test_lr)

val_metrics_ridge = metrics(y_val, yhat_val_ridge)
test_metrics_ridge = metrics(y_test, yhat_test_ridge)

print("\nValidation Metrics - Linear Regression:", val_metrics_lr)
print("Validation Metrics - Optimized Ridge Regression:", val_metrics_ridge)

print("\nTest Metrics - Linear Regression:", test_metrics_lr)
print("Test Metrics - Optimized Ridge Regression:", test_metrics_ridge)

# ========================
# Plot Actual vs Predicted
# ========================
def plot_actual_vs_pred(y_true, preds_dict, title="Actual vs Predicted (Test Set)"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Actual", linewidth=2)
    for name, preds in preds_dict.items():
        plt.plot(y_true.index, preds, label=name, alpha=0.8)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Close Price")
    plt.legend(); plt.show()

preds_test = {
    "Linear Regression": yhat_test_lr,
    "Optimized Ridge Regression": yhat_test_ridge
}
plot_actual_vs_pred(y_test, preds_test, title="Actual vs Predicted (Test)")
