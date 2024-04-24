import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
import warnings

sp500 = yf.download('^GSPC', start="1990-01-01")
sp500['Log Returns'] = np.log(sp500['Adj Close'] - np.log(sp500['Adj Close'].shift(1)))
sp500['Log Returns'].dropna(inplace=True)
sp500.dropna(inplace=True)


# train_data, test_data = sp500['Log Returns'][]

def plot_price(ticker):
    plt.figure(figsize=(10,6))
    plt.plot(ticker.index, ticker['Adj Close'], label='Price')
    plt.ylabel('Price')
    plt.xlabel('Date')
    plt.title('SP500 Price Chart')
    plt.legend()
    plt.show()

def plot_logs(ticker):
    plt.figure(figsize=(10,6))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.ylabel('Log Returns')
    plt.xlabel('Date')
    plt.title('SP500 Log Returns')
    plt.legend()
    plt.show()


def decompose_data(ticker):
    decompose_data = seasonal_decompose(ticker['Log Returns'].dropna(), model = 'additive', period=365, extrapolate_trend='freq')
    daily_frequency = ticker.asfreq(freq='D')
    decompose_data.plot()
    plt.show()

def stationarity_test(ticker):
    ticker['Log Returns'] = ticker['Log Returns'].diff().diff()
    dftest = adfuller(ticker['Log Returns'].dropna(), autolag = 'AIC')
    print("1. ADF : ", dftest[0])
    print("2. P-Value : ", dftest[1])
    print("3. Num Of Lags : ", dftest[2])
    print("4. Num Of Observations Used For ADF Regression and Critical Values Calculation :", dftest[3])
    print("5. Critical Values :")
    for key, val in dftest[4].items():
        print("\t",key, ": ", val)

def acf_plot(ticker):
    fig, ax = plt.subplots(figsize=(12,8))
    plot_acf(ticker['Log Returns'].diff().dropna(), lags=40, alpha=0.05, ax=ax)
    ax.set_title('Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    plt.show()

def pacf_plot(ticker):
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_pacf(ticker['Log Returns'].diff().dropna(), lags=40, alpha=0.05, ax=ax)
    ax.set_title('Partial Autocorrelation Function with 95% Confidence Interval')
    ax.set_xlabel('Lag')
    ax.set_ylabel('Partial Autocorrelation')
    plt.show()


def arima_model(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    ticker['forecast log returns'] = model_fit.predict(start=8613, end=8643)
    forecast = model_fit.get_forecast(steps=30)
    ci = forecast.conf_int(alpha=0.05)
    plt.figure(figsize=(10, 6))
    plt.plot(ticker.index, ticker['Log Returns'], label='Log Returns')
    plt.plot(test.index, test, label='Actual Log Returns')
    plt.plot(ci.iloc[:, 0], label='Lower CI', color='gray')
    plt.plot(ci.iloc[:, 1], label='Upper CI', color='gray')
    plt.plot(ticker['forecast log returns'], label='Forecasted Log Returns')
    plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color='gray', alpha=0.25)
    plt.title('ARIMA Forecasted Log Returns vs Actual Log Returns with 95% Confidence Interval')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.show()

train_data, test_data = sp500['Log Returns'][:8613], sp500['Log Returns'][8612:]
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
# auto_arima = pm.auto_arima(sp500['Log Returns'], seasonal=False, stepwise=False, only_arima=True, start_p=0, start_d=1, start_q=0, max_p=5, max_d=5, max_q=5)
# print(auto_arima.summary())


def model_arima(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast_series = model_fit.forecast(30, alpha=0.05)
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)

    plt.figure(figsize=(12,6))
    train.plot(color='blue', label='Training Data (Log Returns)')
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower CI', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper CI', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('Forecasted Log Returns vs Actual Log Returns SP500 with 95% Confidence Interval')
    plt.show()


def model_close_arima(ticker, train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    ci = forecast.conf_int(alpha=0.05)
    forecast_series = model_fit.forecast(30, alpha=0.05)
    plt.figure(figsize=(12,6))
    test.plot(color='green', label='Actual Log Returns')
    plt.plot(test.index, forecast_series, label='forecast', color='red')
    plt.plot(test.index, ci.iloc[:, 0], label='Lower CI', color='gray')
    plt.plot(test.index, ci.iloc[:, 1], label='Upper CI', color='gray')
    plt.legend()
    plt.xlabel('Date')
    plt.ylabel('Log Returns')
    plt.title('Forecasted Log Returns vs Actual Log Returns SP500 with 95% Confidence Interval')
    plt.show()

def arima_diagnostics(ticker, train, test, params):
    arima_model = ARIMA(train, order=params)
    arima_model_fit = arima_model.fit()
    arima_model_fit.plot_diagnostics(figsize=(12, 8))
    plt.show()
    residuals = arima_model_fit.resid
    Btest = acorr_ljungbox(residuals, lags=20, return_df=True)

    print(Btest)


def mae_rmse(train, test, params):
    model = ARIMA(train, order=params)
    model_fit = model.fit()
    forecast = model_fit.get_forecast(30)
    rmse = mean_squared_error(test, forecast.predicted_mean, squared=False)
    mae = mean_absolute_error(test, forecast.predicted_mean)
    print(f'MAE: {mae}\nRMSE: {rmse}')


mae_rmse(train_data, test_data, (0,1,5))

# model_close_arima(sp500, train_data, test_data, (0,1,5))
# model_arima(sp500, train_data, test_data, (0,1,5))
# arima_diagnostics(sp500, train_data, test_data, (0,1,5))

# model_arima(sp500, train_data, test_data, (0,1,5))
# model_arima_forecast(sp500, (0,1,5))