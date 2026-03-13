import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

START = "2023-01-01"
END   = "2025-01-01"
TRAIN_FRAC = 0.8
N = 60

OUT_FIG = "lstmch3fin.png"

data = yf.download("BTC-USD", start=START, end=END, interval="1d")[["Close"]].dropna()
prices = data["Close"]

split_idx = int(len(prices) * TRAIN_FRAC)
train = prices.iloc[:split_idx]
test  = prices.iloc[split_idx:]

TEST_START = test.index.min()
TEST_END   = test.index.max()

scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled  = scaler.transform(test.values.reshape(-1, 1))

def create_sequences(arr2d, window):
    X, y = [], []
    for i in range(window, len(arr2d)):
        X.append(arr2d[i-window:i, 0])
        y.append(arr2d[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, N)
X_test,  y_test  = create_sequences(test_scaled,  N)

X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test  = X_test.reshape((X_test.shape[0],  X_test.shape[1],  1))

model = Sequential([
    LSTM(64, return_sequences=False, input_shape=(N, 1)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse")

early_stop = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.1,
    shuffle=False,
    callbacks=[early_stop],
    verbose=1
)

y_pred_scaled = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred_scaled).reshape(-1)
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(-1)

effective_start = (TEST_START + pd.Timedelta(days=N)).date()

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))

print("\n Price-Only LSTM (Real BTC Test Window):")
print(f"Shared test window: {TEST_START.date()} → {TEST_END.date()}")
print(f"LSTM evaluation begins: {effective_start} (lookback N={N})")
print(f"MAE:  {mae:.2f} USD")
print(f"RMSE: {rmse:.2f} USD")

test_index = np.arange(len(y_true))

plt.figure(figsize=(10, 5))
plt.plot(test_index, y_true, label="Actual BTC Price")
plt.plot(test_index, y_pred, label="Predicted BTC Price")
plt.title("LSTM Prediction vs Actual BTC Price ")
plt.xlabel("Test Time Index (days)")
plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()