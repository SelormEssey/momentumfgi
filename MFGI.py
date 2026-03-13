import os
import json
import random
from urllib.request import urlopen

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

#setup
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CSV_PATH = "bitcoin_sentiments_21_24.csv"
OUT_DIR = "MFGIFIGS2"
ALT_DIR = "data"
ALT_CSV_PATH = os.path.join(ALT_DIR, "alternative_me_fgi.csv")
ALT_API_JSON = "https://api.alternative.me/fng/?limit=0"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ALT_DIR, exist_ok=True)

Q_LOW = 0.33
Q_HIGH = 0.66

W_MOM = 0.40
W_VOL = 0.30
W_SENT = 0.30
ROLL = 30
LSTM_WINDOW = 60


#helper functions
def expanding_minmax_to_100(series: pd.Series) -> pd.Series:
    mn = series.expanding().min()
    mx = series.expanding().max()
    scaled = (series - mn) / (mx - mn)
    return (scaled * 100).replace([np.inf, -np.inf], np.nan).fillna(50.0)


def save_table_png(df: pd.DataFrame, path: str, title: str = "", fontsize: int = 10):
    df_show = df.copy()
    for c in df_show.columns:
        df_show[c] = df_show[c].astype(str)

    fig_w = min(18, max(10, len(df_show.columns) * 2.2))
    fig_h = min(12, max(3, 0.5 * (len(df_show) + 1)))

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    table = ax.table(
        cellText=df_show.values,
        colLabels=df_show.columns,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(1.0, 1.25)

    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")

    if title:
        ax.set_title(title, pad=12)

    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def load_alternative_me_fgi(csv_path: str = ALT_CSV_PATH, fetch_if_missing: bool = True) -> pd.DataFrame:
    if (not os.path.exists(csv_path)) and fetch_if_missing:
        with urlopen(ALT_API_JSON) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        if "data" not in payload:
            raise ValueError("Alternative.me JSON payload missing 'data'.")
        raw = pd.DataFrame(payload["data"])
        raw.to_csv(csv_path, index=False)
    else:
        raw = pd.read_csv(csv_path)

    raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
    raw["timestamp"] = pd.to_numeric(raw["timestamp"], errors="coerce")
    raw["Date"] = pd.to_datetime(raw["timestamp"], unit="s", errors="coerce").dt.normalize()

    out = (
        raw.rename(columns={"value": "AFGI"})[["Date", "AFGI"]]
        .dropna(subset=["Date", "AFGI"])
        .drop_duplicates(subset=["Date"], keep="last")
        .sort_values("Date")
        .reset_index(drop=True)
    )
    return out


def make_sequences(data_2d: np.ndarray, window: int, target_col: int = 0):
    X, y = [], []
    for i in range(window, len(data_2d)):
        X.append(data_2d[i - window:i, :])
        y.append(data_2d[i, target_col])
    return np.array(X), np.array(y)


def train_eval_lstm(name: str, df_feat: pd.DataFrame, feature_cols: list[str], window: int = 60):
    assert feature_cols[0] == "Close", "feature_cols must start with 'Close'"

    work = df_feat[["Date"] + feature_cols].dropna().copy()
    data = work[feature_cols].values.astype(float)
    dates = pd.to_datetime(work["Date"]).reset_index(drop=True)

    train_size = int(len(data) * 0.8)
    train = data[:train_size]
    test = data[train_size:]

    scaler_close = MinMaxScaler((0, 1))
    train_close = scaler_close.fit_transform(train[:, [0]])
    test_close = scaler_close.transform(test[:, [0]])

    if data.shape[1] > 1:
        scaler_other = MinMaxScaler((0, 1))
        train_other = scaler_other.fit_transform(train[:, 1:])
        test_other = scaler_other.transform(test[:, 1:])
        train_scaled = np.hstack([train_close, train_other])
        test_scaled = np.hstack([test_close, test_other])
    else:
        train_scaled = train_close
        test_scaled = test_close

    X_train, y_train = make_sequences(train_scaled, window, target_col=0)

    test_input = np.vstack([train_scaled[-window:], test_scaled])
    X_test, y_test = make_sequences(test_input, window, target_col=0)

    test_dates = dates.iloc[train_size:].reset_index(drop=True)

    print("\n" + "=" * 70)
    print(name)
    print("Features:", feature_cols)
    print("Rows used:", len(work))
    print("Train size:", train_size)
    print("Test size:", len(test))
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
    print("len(test_dates):", len(test_dates))
    print("len(y_test):", len(y_test))
    print("=" * 70)

    model = Sequential([
        Input(shape=(window, X_train.shape[2])),
        LSTM(64),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])

    es = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        shuffle=False,
        callbacks=[es],
        verbose=1
    )

    y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1, 1)

    y_pred = scaler_close.inverse_transform(y_pred_scaled).ravel()
    y_true = scaler_close.inverse_transform(y_test.reshape(-1, 1)).ravel()

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))

    print(f"{name} RESULTS")
    print(f"MAE:  {mae:.2f} USD")
    print(f"RMSE: {rmse:.2f} USD")
    print(f"MSE:  {mse:.2f} USD^2")

    pred_df = pd.DataFrame({
        "Date": test_dates,
        "Actual": y_true,
        "Predicted": y_pred
    })

    print(f"\n{name} date gaps:")
    print(pred_df["Date"].diff().value_counts().head(10))

    plt.figure(figsize=(11, 4.5))
    plt.plot(test_dates, y_true, label="Actual")
    plt.plot(test_dates, y_pred, label="Predicted")
    plt.title(f"{name}: Predicted vs Actual BTC Close")
    plt.xlabel("Date")
    plt.ylabel("BTC Close (USD)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            OUT_DIR,
            f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')}_pred_vs_actual.png"
        ),
        dpi=300
    )
    plt.close()

    return {
        "model": name,
        "mae": mae,
        "rmse": rmse,
        "mse": mse,
        "pred_df": pred_df,
        "history": history.history
    }

#loading and preprocessing sentiment data
raw = pd.read_csv(CSV_PATH)
raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
raw = raw.dropna(subset=["Date", "Accurate Sentiments"]).copy()
raw["Date"] = raw["Date"].dt.normalize()

daily = (
    raw.groupby("Date")
       .agg(
           sentiment_raw=("Accurate Sentiments", "mean"),
           headlines=("Accurate Sentiments", "size")
       )
       .reset_index()
       .sort_values("Date")
       .reset_index(drop=True)
)

#force full daily calendar so plots do not bridge missing dates
full_dates = pd.date_range(start=daily["Date"].min(), end=daily["Date"].max(), freq="D")
daily = (
    daily.set_index("Date")
         .reindex(full_dates)
         .rename_axis("Date")
         .reset_index()
)

daily["sentiment_raw"] = daily["sentiment_raw"].ffill()
daily["headlines"] = daily["headlines"].fillna(0)

daily.to_csv(os.path.join(OUT_DIR, "daily_sentiment_processed.csv"), index=False)

first_date = daily["Date"].min()
last_date = daily["Date"].max()

print("\nSENTIMENT PIPELINE")
print("Sentiment range:", first_date.date(), "to", last_date.date())
print("Number of daily rows:", len(daily))
print(daily.head())

plt.figure(figsize=(12, 4))
plt.plot(daily["Date"], daily["sentiment_raw"])
plt.title("Daily Mean News Sentiment")
plt.xlabel("Date")
plt.ylabel("Daily mean sentiment")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step1_daily_sentiment.png"), dpi=300)
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(daily["sentiment_raw"], bins=40, edgecolor="black")
plt.title("Distribution of Daily Mean Sentiment")
plt.xlabel("Daily mean sentiment")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step2_sentiment_histogram.png"), dpi=300)
plt.close()


#discretization + sentiment lags
low_q = daily["sentiment_raw"].quantile(Q_LOW)
high_q = daily["sentiment_raw"].quantile(Q_HIGH)

daily["sentiment_disc"] = np.where(
    daily["sentiment_raw"] >= high_q, 1,
    np.where(daily["sentiment_raw"] <= low_q, -1, 0)
)

print("\nDISCRETIZATION")
print("33rd percentile:", low_q)
print("66th percentile:", high_q)
print(daily["sentiment_disc"].value_counts().sort_index())

counts = daily["sentiment_disc"].value_counts().reindex([-1, 0, 1]).fillna(0).astype(int)

plt.figure(figsize=(6, 4))
plt.bar(["-1 (Fear)", "0 (Neutral)", "+1 (Greed)"], counts.values, edgecolor="black")
plt.title("Discretized Sentiment Class Balance")
plt.ylabel("Number of days")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step3_discretized_class_balance.png"), dpi=300)
plt.close()

for k in [1, 2, 3]:
    daily[f"sentiment_disc_lag{k}"] = daily["sentiment_disc"].shift(k)
    daily[f"sentiment_raw_lag{k}"] = daily["sentiment_raw"].shift(k)

lag_preview = daily[[
    "Date", "sentiment_raw", "sentiment_disc",
    "sentiment_disc_lag1", "sentiment_disc_lag2", "sentiment_disc_lag3",
    "sentiment_raw_lag1", "sentiment_raw_lag2", "sentiment_raw_lag3"
]].dropna().head(12).copy()

lag_preview["sentiment_raw"] = lag_preview["sentiment_raw"].map(lambda x: f"{x:.4f}")
lag_preview["sentiment_raw_lag1"] = lag_preview["sentiment_raw_lag1"].map(lambda x: f"{x:.4f}")
lag_preview["sentiment_raw_lag2"] = lag_preview["sentiment_raw_lag2"].map(lambda x: f"{x:.4f}")
lag_preview["sentiment_raw_lag3"] = lag_preview["sentiment_raw_lag3"].map(lambda x: f"{x:.4f}")

save_table_png(
    lag_preview,
    os.path.join(OUT_DIR, "step4_lagged_sentiment_table.png"),
    title="Lagged Sentiment Features (Example Rows)"
)


#pullbtc and align with sentiment
start = (first_date - pd.Timedelta(days=90)).date()
end = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

btc = yf.download("BTC-USD", start=str(start), end=end, interval="1d", progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc.columns = btc.columns.get_level_values(0)

btc = btc[["Close"]].dropna().reset_index()
btc["Date"] = pd.to_datetime(btc["Date"], errors="coerce").dt.normalize()

# Now merged uses full daily sentiment calendar
merged = pd.merge(
    btc,
    daily,
    on="Date",
    how="left"
).sort_values("Date").reset_index(drop=True)

# Fill any leading sentiment gaps after merge
merged["sentiment_raw"] = merged["sentiment_raw"].ffill().bfill()
merged["headlines"] = merged["headlines"].fillna(0)
merged["sentiment_disc"] = merged["sentiment_disc"].ffill().bfill()

for k in [1, 2, 3]:
    merged[f"sentiment_disc_lag{k}"] = merged[f"sentiment_disc_lag{k}"].ffill().bfill()
    merged[f"sentiment_raw_lag{k}"] = merged[f"sentiment_raw_lag{k}"].ffill().bfill()

print("\nBTC + SENTIMENT MERGE")
print("Rows after merge:", len(merged))
print("Merged range:", merged["Date"].min().date(), "to", merged["Date"].max().date())
print(merged[["Date", "Close", "sentiment_raw", "sentiment_disc"]].head())

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(merged["Date"], merged["Close"], label="BTC Close")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Close (USD)")
ax1.set_title("BTC Close vs Daily News Sentiment")

ax2 = ax1.twinx()
ax2.plot(merged["Date"], merged["sentiment_raw"], alpha=0.35, label="Daily sentiment")
ax2.set_ylabel("Daily mean sentiment")

l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step5_price_vs_sentiment.png"), dpi=300)
plt.close()


#MFGI Building
df = merged.copy()

df["MA_30"] = df["Close"].rolling(ROLL).mean()
df["Momentum"] = df["Close"] / df["MA_30"]

df["LogReturn"] = np.log(df["Close"]).diff()
df["Volatility"] = df["LogReturn"].rolling(ROLL).std()

df = df.dropna().reset_index(drop=True)

df["Momentum_Score"] = expanding_minmax_to_100(df["Momentum"])
df["Volatility_Score"] = 100 - expanding_minmax_to_100(df["Volatility"])
df["Sentiment_Score"] = expanding_minmax_to_100(df["sentiment_raw"])

df["MFGI"] = (
    W_MOM * df["Momentum_Score"] +
    W_VOL * df["Volatility_Score"] +
    W_SENT * df["Sentiment_Score"]
).round(2)

for k in [1, 2, 3]:
    df[f"MFGI_lag{k}"] = df["MFGI"].shift(k)

mfgi_preview = df[[
    "Date", "Close", "Momentum_Score", "Volatility_Score", "Sentiment_Score", "MFGI",
    "MFGI_lag1", "MFGI_lag2", "MFGI_lag3"
]].dropna().head(12).copy()

for c in ["Close", "Momentum_Score", "Volatility_Score", "Sentiment_Score", "MFGI", "MFGI_lag1", "MFGI_lag2", "MFGI_lag3"]:
    mfgi_preview[c] = mfgi_preview[c].map(lambda x: f"{x:.2f}")

save_table_png(
    mfgi_preview,
    os.path.join(OUT_DIR, "step6_mfgi_table_preview.png"),
    title="MFGI and Lagged MFGI Features (Example Rows)"
)

print("\nMFGI BUILT")
print(df[["Date", "Momentum_Score", "Volatility_Score", "Sentiment_Score", "MFGI"]].head())

fig, ax1 = plt.subplots(figsize=(12, 5))
ax1.plot(df["Date"], df["Close"], label="BTC Close")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Close (USD)")
ax1.set_title("BTC Close vs Momentum Fear and Greed Index")

ax2 = ax1.twinx()
ax2.plot(df["Date"], df["MFGI"], alpha=0.35, label="MFGI")
ax2.set_ylabel("MFGI (0-100)")
ax2.set_ylim(0, 100)

l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lab1 + lab2, loc="upper left")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step7_price_vs_mfgi.png"), dpi=300)
plt.close()

plt.figure(figsize=(12, 4))
plt.plot(df["Date"], df["MFGI"])
plt.title("Momentum Fear and Greed Index")
plt.xlabel("Date")
plt.ylabel("MFGI (0-100)")
plt.ylim(0, 100)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "step8_mfgi_only.png"), dpi=300)
plt.close()

df.to_csv(os.path.join(OUT_DIR, "model_frame_with_mfgi.csv"), index=False)


# ============================================================
# 6) OPTIONAL ALTERNATIVE.ME OVERLAY
# ============================================================
try:
    alt = load_alternative_me_fgi()
    overlay = (
        pd.merge(df[["Date", "MFGI"]], alt, on="Date", how="inner")
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print("\nALTERNATIVE.ME OVERLAY")
    print("Overlay rows:", len(overlay))
    if len(overlay) > 0:
        print("Overlay range:", overlay["Date"].min().date(), "to", overlay["Date"].max().date())

        plt.figure(figsize=(12, 5))
        plt.plot(overlay["Date"], overlay["MFGI"], label="Momentum MFGI", linewidth=1.6)
        plt.plot(overlay["Date"], overlay["AFGI"], label="Alternative.me FGI", linewidth=1.2, alpha=0.85)
        plt.title("Overlay: Momentum MFGI vs Alternative.me Fear and Greed Index")
        plt.xlabel("Date")
        plt.ylabel("FGI (0-100)")
        plt.ylim(0, 100)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "step9_mfgi_vs_alternative_me.png"), dpi=300)
        plt.close()

        overlay.to_csv(os.path.join(OUT_DIR, "overlay_mfgi_vs_alternative_me.csv"), index=False)
except Exception as e:
    print("\nAlternative.me overlay skipped:", e)

#Dataset for LSTM with MFGI lags
lstm_df = df[[
    "Date", "Close",
    "MFGI_lag1", "MFGI_lag2", "MFGI_lag3"
]].dropna().reset_index(drop=True)

print("\nLSTM DATAFRAME")
print("Rows:", len(lstm_df))
print(lstm_df.head())


#train LSTM with price only
res_price = train_eval_lstm(
    name="Price-only LSTM",
    df_feat=lstm_df,
    feature_cols=["Close"],
    window=LSTM_WINDOW
)


#train LSTM with MFGI lags
res_mfgi = train_eval_lstm(
    name="Hybrid LSTM (Price + MFGI lags)",
    df_feat=lstm_df,
    feature_cols=["Close", "MFGI_lag1", "MFGI_lag2", "MFGI_lag3"],
    window=LSTM_WINDOW
)


#final comparison
metrics = pd.DataFrame([
    {
        "Model": res_price["model"],
        "MAE": round(res_price["mae"], 2),
        "RMSE": round(res_price["rmse"], 2),
        "MSE": round(res_price["mse"], 2)
    },
    {
        "Model": res_mfgi["model"],
        "MAE": round(res_mfgi["mae"], 2),
        "RMSE": round(res_mfgi["rmse"], 2),
        "MSE": round(res_mfgi["mse"], 2)
    }
])

print("\nFINAL MODEL COMPARISON")
print(metrics)

metrics.to_csv(os.path.join(OUT_DIR, "final_model_comparison.csv"), index=False)
save_table_png(
    metrics,
    os.path.join(OUT_DIR, "final_model_comparison_table.png"),
    title="Price-only LSTM vs Hybrid LSTM (Price + MFGI)"
)

print("\nPrediction row counts")
print("Price-only rows:", len(res_price["pred_df"]))
print("Hybrid rows:", len(res_mfgi["pred_df"]))

comparison = pd.merge(
    res_price["pred_df"].rename(columns={
        "Actual": "Actual_PriceOnly",
        "Predicted": "Predicted_PriceOnly"
    }),
    res_mfgi["pred_df"].rename(columns={
        "Actual": "Actual_Hybrid",
        "Predicted": "Predicted_Hybrid"
    }),
    on="Date",
    how="inner"
).sort_values("Date").reset_index(drop=True)

comparison["Actual"] = comparison["Actual_PriceOnly"]

print("\nComparison dataframe checks")
print(comparison.head())
print("Rows in comparison:", len(comparison))
print("Dates sorted:", comparison["Date"].is_monotonic_increasing)
print("Date gaps:")
print(comparison["Date"].diff().value_counts().head(10))

comparison.to_csv(os.path.join(OUT_DIR, "final_predictions_comparison.csv"), index=False)

plt.figure(figsize=(12, 5))
plt.plot(
    res_price["pred_df"]["Date"],
    res_price["pred_df"]["Actual"],
    label="Actual",
    linewidth=2
)
plt.plot(
    res_price["pred_df"]["Date"],
    res_price["pred_df"]["Predicted"],
    label="Price-only LSTM"
)
plt.plot(
    res_mfgi["pred_df"]["Date"],
    res_mfgi["pred_df"]["Predicted"],
    label="Hybrid LSTM (Price + MFGI)"
)
plt.title("Final Comparison: Price-only LSTM vs Hybrid LSTM with MFGI")
plt.xlabel("Date")
plt.ylabel("BTC Close (USD)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_lstm_comparison_plot.png"), dpi=300)
plt.close()

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))
plt.bar(x - width / 2, metrics["RMSE"], width, label="RMSE")
plt.bar(x + width / 2, metrics["MAE"], width, label="MAE")
plt.xticks(x, metrics["Model"], rotation=10)
plt.ylabel("Error (USD)")
plt.title("Error Comparison Across LSTM Variants")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "final_error_bar_chart.png"), dpi=300)
plt.close()

print("\nSaved outputs to:", OUT_DIR)
print("Files:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" -", f)