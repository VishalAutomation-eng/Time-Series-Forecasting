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
# Add multiple moving averages
for window in [5, 10, 20, 50, 100, 200]:
    tesla_data[f"SMA_{window}"] = tesla_data["Close"].rolling(window).mean()

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

# Lags of Close (1..5 days)
for lag in range(1, 6):
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
feature_cols = [
    "Close_lag_1","Close_lag_2","Close_lag_3","Close_lag_4","Close_lag_5",
    "Ret_lag_1","Ret_lag_2","Ret_lag_3","Ret_lag_4","Ret_lag_5",
    "roll_mean_5","roll_mean_10","roll_std_5","roll_std_10","roll_min_14","roll_max_14",
    "Vol_lag_1","Vol_sma_5",
    "dayofweek","month"
]

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
print(f"Date ranges -> Train: {X_train.index.min().date()} to {X_train.index.max().date()}, "
      f"Val: {X_val.index.min().date()} to {X_val.index.max().date()}, "
      f"Test: {X_test.index.min().date()} to {X_test.index.max().date()}")

# ========================
# Train Linear Regression Model
# ========================
lr_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("lr", LinearRegression())
])
lr_pipe.fit(X_train, y_train)

# ========================
# Train Ridge Regression Model
# ========================
ridge_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])
ridge_pipe.fit(X_train, y_train)

# Predictions
yhat_val_lr  = lr_pipe.predict(X_val)
yhat_test_lr = lr_pipe.predict(X_test)

yhat_val_ridge  = ridge_pipe.predict(X_val)
yhat_test_ridge = ridge_pipe.predict(X_test)

# Metrics
val_metrics_lr = metrics(y_val, yhat_val_lr)
test_metrics_lr = metrics(y_test, yhat_test_lr)

val_metrics_ridge = metrics(y_val, yhat_val_ridge)
test_metrics_ridge = metrics(y_test, yhat_test_ridge)

print("\nValidation Metrics - Linear Regression:", val_metrics_lr)
print("Validation Metrics - Ridge Regression:", val_metrics_ridge)

print("\nTest Metrics - Linear Regression:", test_metrics_lr)
print("Test Metrics - Ridge Regression:", test_metrics_ridge)

# ========================
# Plot Actual vs Predicted
# ========================
def plot_actual_vs_pred(y_true, preds_dict, title="Actual vs Predicted (Test Set)"):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Actual")
    for name, preds in preds_dict.items():
        plt.plot(y_true.index, preds, label=name)
    plt.title(title)
    plt.xlabel("Date"); plt.ylabel("Close Price")
    plt.legend(); plt.show()

preds_test = {
    "Linear Regression": yhat_test_lr,
    "Ridge Regression": yhat_test_ridge
}
plot_actual_vs_pred(y_test, preds_test, title="Actual vs Predicted (Test)")
