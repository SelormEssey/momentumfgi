# momentumfgi

Momentum FGI is a cryptocurrency forecasting project that compares Naive, ARIMA, and LSTM models while incorporating sentiment-aware features into a transparent Fear and Greed Index framework.

## Project Overview

This project explores whether sentiment-aware indicators can improve short-horizon cryptocurrency forecasting and interpretation. It combines traditional forecasting models with a custom Momentum Fear and Greed Index built from price behavior and sentiment-related features.

## Files

- `naive.py` – Naive baseline forecasting model
- `arima.py` – ARIMA forecasting model
- `lstm.py` – LSTM forecasting model
- `MFGI.py` – Momentum Fear and Greed Index pipeline
- `bitcoin_sentiments_21_24.csv` – sentiment dataset
- `data/` – input datasets
- `figures/` – core output figures
- `MFGIFIGS2/` – additional generated figures, tables, and comparison outputs

## Goal

To evaluate whether transparent sentiment-aware indicators improve short-horizon crypto forecasting and interpretation.

## Models Used

- Naive baseline
- ARIMA
- LSTM
- Sentiment-enhanced forecasting through MFGI

## Outputs

The project generates:
- forecast comparison plots
- sentiment preprocessing visuals
- Fear and Greed Index charts
- model comparison tables
- prediction result files

