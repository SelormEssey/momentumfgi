import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from itertools import product
import warnings
warnings.filterwarnings("ignore")

START = "2023-01-01"
END   = "2025-01-01"
TRAIN_FRAC = 0.8
FUTURE_DAYS = 15

OUT_FIG = "figures/arimach3fin.png"


data = yf.download("BTC-USD", start=START, end=END, interval="1d")[["Close"]].dropna()
series = data["Close"]


split_idx = int(len(series) * TRAIN_FRAC)
train = series.iloc[:split_idx]
test  = series.iloc[split_idx:]

TEST_START = test.index.min()
TEST_END   = test.index.max()


p_values = range(0, 4)
d_values = range(0, 2)
q_values = range(0, 4)

best_order, best_mse, best_model = None, float("inf"), None

for p, d, q in product(p_values, d_values, q_values):
    order = (p, d, q)
    try:
        model = ARIMA(train, order=order).fit()
        preds = model.forecast(steps=len(test))
        mse = np.mean((test.values - preds.values) ** 2)
        if mse < best_mse:
            best_mse = mse
            best_order = order
            best_model = model
    except:
        continue


forecast_all = best_model.forecast(steps=len(test) + FUTURE_DAYS)
test_preds = forecast_all.iloc[:len(test)]
future_preds = forecast_all.iloc[len(test):]

future_index = pd.date_range(start=TEST_END, periods=FUTURE_DAYS + 1, freq="D")[1:]


mae = mean_absolute_error(test.values, test_preds.values)
rmse = np.sqrt(mean_squared_error(test.values, test_preds.values))

print("\n ARIMA (Real BTC Test Window):")
print(f"Test window: {TEST_START.date()} → {TEST_END.date()}")
print(f"Best ARIMA order: {best_order}")
print(f"MAE:  {mae:.2f} USD")
print(f"RMSE: {rmse:.2f} USD")


plt.figure(figsize=(10, 5))
plt.plot(test.index, test.values, label="Actual BTC", color="steelblue")
plt.plot(test.index, test_preds.values, label="Test Predictions", color="orange")
plt.plot(future_index, future_preds.values, label="15-Day Forecast", color="red")
plt.axvline(test.index[0], color="gray", linestyle="--", label="Train/Test Split")

plt.title(f"ARIMA Forecast (Best Model: {best_order}) — Test Window")
plt.xlabel("Date")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()
