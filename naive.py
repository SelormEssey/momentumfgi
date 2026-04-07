import os
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

START = "2023-01-01"
END = "2025-01-01"
TRAIN_FRAC = 0.8

OUT_FIG = "figures/naivech3fin.png"
os.makedirs("figures", exist_ok=True)

data = yf.download("BTC-USD", start=START, end=END, interval="1d")[["Close"]].dropna()
series = data["Close"]

split_idx = int(len(series) * TRAIN_FRAC)
train = series.iloc[:split_idx]
test = series.iloc[split_idx:]

TEST_START = test.index.min()
TEST_END = test.index.max()

naive_forecast = series.shift(1).loc[TEST_START:TEST_END].copy()
naive_forecast.iloc[0] = train.iloc[-1]

mae = mean_absolute_error(test.values, naive_forecast.values)
rmse = np.sqrt(mean_squared_error(test.values, naive_forecast.values))

print("\nNaive Baseline")
print(f"Test window: {TEST_START.date()} to {TEST_END.date()}")
print(f"MAE: {mae:.2f} USD")
print(f"RMSE: {rmse:.2f} USD")

plt.figure(figsize=(10, 5))
plt.plot(test.index, test.values, label="Actual BTC Closing Prices", color="black")
plt.plot(test.index, naive_forecast.values, label="Naive forecast", linestyle="--")
plt.title("Naive Forecast vs Actual BTC Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
plt.show()