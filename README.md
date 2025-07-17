# Numerai ML Signal Project

This repo contains a full machine learning pipeline I developed for the [Numerai Signals tournament](https://numer.ai).

## 🧠 Objective

The goal is to generate predictive signals for financial assets using machine learning models trained on time-lagged feature data. Models are evaluated using **Spearman correlation** with targets and ranked weekly.

## 📊 Methods

- LightGBM Regressor with custom lag features
- Feature interactions & filtering based on signal quality
- Temporal validation strategy using era-aware splits
- Multiple targets (factor-neutral and raw return)
- Spearman-based model selection and ensembling

## 📁 Structure

- `notebooks/`: Feature engineering, training, evaluation
- `src/`: Pipeline and reusable model functions
- `submissions/`: Final model signal exports
- `requirements.txt`: Reproducible environment

## 🔥 Results

Top Spearman scores (validation):
- `target_factor_neutral_20`: **0.0485**
- `target_raw_return_20`: **0.0314**
- `target_raw_return_60`: **0.0264**

## 🚀 Skills Used

- Python, pandas, numpy, LightGBM
- Spearman scoring, AUC analysis
- Financial time series
- Leak prevention and feature lagging
- Automation for signal generation

## 📫 Contact

*Jordan Lloyd*  
MS Electrical Engineering @ UT Arlington  
'www.linkedin.com/in/jordanlloydtamu'
