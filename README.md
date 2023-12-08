# Timeseries Transformer

This repository is created for the Master Thesis "Transformer based model for time-series within Finance" at the University of Copenhagen 2023.

The repository holds three main Jupyter Notebooks:
- Data Preparation
- Model training
- Forecast and portfolio

Moreover, the repository holds three Python files with code used for the three main Jupyter Notebooks:
- Timeserie_models
- Positional_encoder
- utils

#### Data
Raw data and data prepared as input to the time-series models are collected in the folder 'Data'.

#### Output
Outputs from the individual models (training and forecasts) and the portfolio creation are collected in the folder 'Models'.


### Data Preparation
This Jyputer Notebook reads in the raw data, preprocess it creates the sequences ready to be fed into the models. It does this for both the training and test data.

### Model Training
This Jupyter Notebook reads in the preprocessed data and trains the individual models while saving the important parameter values.

*Notice: This program uses a GPU for training and does take quite some time to run. It is recommended to use the Colab runtime-type 'V100' for this program, but the runtime-type 'T4' should also work.*

### Forecasting and portfolio
This Jupyter Notebook uses the best parameter values from the train models and creates a forecast on the test data (SPX). Afterwards, it makes the portfolios.

*Notice: This program uses a GPU and requires a high amount of RAM since it takes all the test data into the model in one go. Therefore it is recommended to use the Colab runtime-type 'A100' if possible, however the runtime-type 'V100' is also an option but it might crash. If the program crashes with 'V100' just shut down the program and try again (the 'V100' runtime-type is on the edge of what is needed to run the program and most times it can do so)*


