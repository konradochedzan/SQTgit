# Selected Quantitative Tools

A Python-based toolkit for forecasting and analyzing S&P 500 returns using neural and statistical models. This repository includes model definitions, training pipelines, evaluation scripts, and utilities for saving and result analysis.


## Overview

This project implements a stochastic volatility forecasting framework, using a selection of machine learning models. We can train, evaluate, compare, and persist neural forecasting models on rolling windows of S&P 500 data, perform statistical tests on out-of-sample predictions, and aggregate final results.

## File Structure

- **`Base_models.py`**  
  Defines generic feed-forward & autoencoder architectures, plus an Elastic Net regularization loss.

- **`Simple_models.py`**  
  Contains implementations of the basic models we use: Feed-forward, LSTM, Transformer, and CNN.

- **`TM_models.py`**  
 Contains implementations of more advanced model, TFT and TCN.  


- **`environment.py`**  
  - Data preparation
  - Model training & evaluation  
  - Hyperparameter tuning & architecture selection  
  - Plotting & performance metrics

- **`statistical analysis.py`**  
  Post-training analysis:  
  - Loads out-of-sample forecasts  
  - Computes MAE, RMSE, R², Durbin–Watson, Jarque–Bera, Ljung–Box  
  - Runs Diebold–Mariano pairwise model comparison tests

- **`saving.py`**  
  Persistence utilities:  
  - `save_model` / `load_model` (joblib)  
  - `save_predictions` / `load_predictions` (CSV)

- **`s.py`**  
  Parses summary CSVs of architectures & tuning runs, extracts key statistics, and creates a cleaned `model_results.csv`.

- **`Arch_selection_final_training.py`**  
    - Selects best model architectures (with/without autoencoder)
    - Trains final models
    - Saves trained models and their predictions

