# Time Series Transformer

This repository is created for the Master's Thesis "A Transformer Based Model for Time Series within Finance" at the University of Copenhagen 2023.

Abstract of the paper: 

*The recent development and success of Large Language Models, such as ChatGPT, builds
on the introduction of Transformer models, which was first published in (Vaswani et al.,
2017). The advantages of Transformer models, to retrieve information in long sequences of
data instead of single-pointed input, have recently been shown to be effective for time series
forecasting and to outperform existing models such as AR and RNN models ((Zhou et al.,
2021) and (Li et al., 2019)). Moreover, within the financial field, the Transformer models
have been able to significantly do better for the task of asset volatility prediction than more
classic methods such as GARCH models (Ge, Lalbakhsh, Isai, Lensky, & Suominen, 2023).
This leads to the question of whether Transformer models can effectively be implemented
for asset return predictions and outperform existing methods in a portfolio setting.*

*The paper finds that the Transformer model can capture
valuable information even in relatively volatile time series, such as weekly stock market
returns, which can be implemented effectively in simple long portfolios. The portfolios
created based on the Transformer predictions outperformed a series of benchmark portfolios,
such as three different momentum-based portfolios, an equally weighted portfolio, and an
LSTM forecasting portfolio.
The question of how the Transformer predictions are best implemented in a portfolio setting
is still an open question. However, the paper explores different Loss functions for the specific
task of creating predictions for a portfolio setting.*

## Presentation
The paper creates a model based on the Transformer from the paper by Vaswani et al.,
2017 and alternates it for time series data.

![Image of the Transformer Model](/Images/Transformer_model.png "Transformer model")

The model is trained on the 12% largest stocks (Market Capitalization) in North America and Western Europe for the period 2012 - 2017, which gives approximately 2500 assets. The Transformer model uses weekly returns and is trained with four different loss functions; Mean Squared Error (MSE), Weighted Mean Squared Error (WMSE), Adjusted Mean Squared Error (AdjMSE), and Negative Correlation (NegCorr). Moreover, a standard LSTM model with an MSE loss function is trained as a reference model. 

The five trained models are used to forecast the four-week return of all S&P-500 stocks for the period 2018 - 2023. Based on the model predictions, simple portfolios as created to evaluate the performance of each model in the out-of-sample period. Moreover, the five model-based portfolios are compared to the overall S&P-500 equally weighted index. 

![Image of the Portfolios created in the paper](/Images/Out-of-sample_portfolios.png "Out-of-sample Portfolios")

The file 'Presentation' is a presentation created for the defense of the Master's Thesis and goes through the main findings in the paper.

## Code
All code is written to be run in Google Colab.

The repository holds three main Jupyter Notebooks:
- Data Preparation
- Model training
- Forecast and portfolio

Moreover, the repository holds three Python files with code used within the three main Jupyter Notebooks:
- Transformer_model 
- Positional_encoder
- utils

#### Data
Raw data and data prepared as input to the time-series models are collected in the folder 'Data'.

*Notice: All raw data has been removed for the publication of the repository since I do not have the right to share it.*

#### Output
Outputs from the individual models (training and forecasts) and the portfolio creation are collected in the folder 'Models'.


### Data Preparation
This Jyputer Notebook reads in the raw data, preprocesss it, and creates the sequences ready to be fed into the models. It does this for both the training and test data.

### Model Training
This Jupyter Notebook reads in the preprocessed data and trains the individual models while saving the important parameter values.

### Forecasting and portfolio
This Jupyter Notebook uses the best parameter values from the trained models and creates a forecast on the test data (SPX). Afterwards, it makes the portfolios.

References used for the code:

[1] https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e 

[2] https://github.com/KasperGroesLudvigsen/influenza_transformer

[3] https://towardsdatascience.com/how-to-run-inference-with-a-pytorch-time-series-transformer-394fd6cbe16c

[4] https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3973086 (https://github.com/JDE65/CustomLoss)


