# Momentum FGI 📈

Momentum FGI is a fintech and machine learning project exploring how market data and sentiment can be combined to better understand and forecast short-term cryptocurrency price movements.

The project compares traditional and deep learning forecasting approaches while developing a transparent **Momentum Fear & Greed Index (MFGI)** for interpreting Bitcoin market conditions.

## 💳 Fintech Focus

Financial markets are influenced by more than price history alone. Investor sentiment, volatility, and market momentum can all contribute to short-term behavior.

Momentum FGI explores how these signals can be combined into a transparent market indicator and used alongside forecasting models.

The project focuses on:

- cryptocurrency market forecasting
- financial sentiment analysis
- market momentum and volatility
- quantitative time-series modeling
- interpretable market indicators
- AI applications in fintech

## 📊 Project Overview

The project evaluates several approaches to Bitcoin price forecasting:

- **Naive Forecasting** — baseline for evaluating model performance
- **ARIMA** — statistical time-series forecasting
- **LSTM** — deep learning for sequential financial data
- **Momentum Fear & Greed Index (MFGI)** — a custom market sentiment indicator combining price momentum, volatility, and sentiment signals

The goal is to evaluate whether sentiment-aware financial indicators provide useful information for short-horizon cryptocurrency forecasting and market interpretation.

## 🧠 Momentum Fear & Greed Index

The MFGI is designed as a transparent alternative to black-box market sentiment indicators.

It combines:

- market momentum
- price volatility
- cryptocurrency news sentiment

These signals are transformed into a 0–100 index representing market conditions ranging from **Extreme Fear** to **Extreme Greed**.

## 🛠️ Models & Methods

- Naive baseline forecasting
- ARIMA
- Long Short-Term Memory Networks (LSTM)
- Sentiment analysis
- Financial time-series analysis
- Rolling market indicators
- Model evaluation using MAE and RMSE

## 📁 Repository Structure

- `naive.py` — baseline financial forecasting model
- `arima.py` — ARIMA time-series model
- `lstm.py` — LSTM forecasting experiments
- `MFGI.py` — Momentum Fear & Greed Index pipeline
- `bitcoin_sentiments_21_24.csv` — cryptocurrency sentiment dataset
- `data/` — financial and sentiment datasets
- `figures/` — forecasting and market-analysis visualizations
- `MFGIFIGS2/` — additional model comparisons and generated outputs

## 📈 Outputs

The project produces:

- Bitcoin price forecasts
- model performance comparisons
- financial sentiment visualizations
- Momentum Fear & Greed Index charts
- market condition classifications
- prediction and evaluation results

## 🎯 Goal

The broader goal of Momentum FGI is to explore how **AI, data science, and transparent market analytics can be applied to financial technology**, particularly in highly sentiment-driven markets such as cryptocurrency.
