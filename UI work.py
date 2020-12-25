#!/usr/bin/env python
# coding: utf-8

# In[ ]:


print('Dependencies loading......')

get_ipython().system('pip install pyforest')
from pyforest import *
import datetime, pickle, copy, warnings
get_ipython().system('pip install cryptocompare')
import cryptocompare
import requests
get_ipython().system('pip install plotly')
get_ipython().system('pip install cufflinks')
import plotly.express as px
import plotly.graph_objects as go
from time import time
from pandas import DataFrame, concat
from sklearn import metrics
from sklearn.linear_model import LinearRegression, ElasticNet
from statsmodels.tsa.arima.model import ARIMA 
from statsmodels.tsa.statespace.sarimax import SARIMAX 
import statsmodels.api as sm
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error, r2_score
from math import sqrt
get_ipython().system('pip install ta')
import ta
get_ipython().system('pip install quandl')
import quandl
get_ipython().system('pip install tscv')

print('Complete....')


print('*********************************************************************************')

# Linear model

start = time()
print('Starting program ( hourly data)....')

apiKey = "43b01c420b66888ce4c91b364647600814578c186e8604322152f44c641ebbc1"
url = "https://min-api.cryptocompare.com/data/histohour"

# BTC data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()

btc1 = DataFrame(result['Data'])
btc1['time'] = pd.to_datetime(btc1['time'],unit='s')
btc1.set_index('time',inplace=True)

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1601632800)
}

result = requests.get(url, params=payload).json()

btc2 = DataFrame(result['Data'])
btc2['time'] = pd.to_datetime(btc2['time'],unit='s')
btc2.set_index('time',inplace=True)

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1593572400)
}

result = requests.get(url, params=payload).json()

btc3 = DataFrame(result['Data'])
btc3['time'] = pd.to_datetime(btc3['time'],unit='s')
btc3.set_index('time',inplace=True)

# 4th 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1596571200)
}

result = requests.get(url, params=payload).json()

btc4 = DataFrame(result['Data'])
btc4['time'] = pd.to_datetime(btc4['time'],unit='s')
btc4.set_index('time',inplace=True)

# combining BTC dataframe
com1 = btc2.append(btc1)
com2 = btc3.append(com1)
btc = btc4.append(com2)
# saving btc data set
#btc.to_csv("bitcoin.csv")


# ETH DATA
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()
eth1 = DataFrame(result['Data'])
eth1['time'] = pd.to_datetime(eth1['time'],unit='s')
eth1.set_index('time',inplace=True)

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1601632800)
}

result = requests.get(url, params=payload).json()
eth2 = DataFrame(result['Data'])
eth2['time'] = pd.to_datetime(eth2['time'],unit='s')
eth2.set_index('time',inplace=True)

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1593572400)
}

result = requests.get(url, params=payload).json()
eth3 = DataFrame(result['Data'])
eth3['time'] = pd.to_datetime(eth3['time'],unit='s')
eth3.set_index('time',inplace=True)

# 4th ETH 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1596571200)
}

result = requests.get(url, params=payload).json()

eth4 = DataFrame(result['Data'])
eth4['time'] = pd.to_datetime(eth4['time'],unit='s')
eth4.set_index('time',inplace=True)

# combining BTC dataframe
com1 = eth2.append(eth1)
com2 = eth3.append(com1)
eth = eth4.append(com2)

# saving ETH data set
#eth.to_csv("Ethereum.csv")

# LTC data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000
}
result = requests.get(url, params=payload).json()
ltc1 = DataFrame(result['Data'])
ltc1['time'] = pd.to_datetime(ltc1['time'],unit='s')
ltc1.set_index('time',inplace=True)

# 2nd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1601632800)
}

result = requests.get(url, params=payload).json()
ltc2 = DataFrame(result['Data'])
ltc2['time'] = pd.to_datetime(ltc2['time'],unit='s')
ltc2.set_index('time',inplace=True)

# 3rd 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1593572400)
}

result = requests.get(url, params=payload).json()
ltc3 = DataFrame(result['Data'])
ltc3['time'] = pd.to_datetime(ltc3['time'],unit='s')
ltc3.set_index('time',inplace=True)

# 4th ETH 2000 data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000,
    "toTs": (1596571200)
}

result = requests.get(url, params=payload).json()

ltc4 = DataFrame(result['Data'])
ltc4['time'] = pd.to_datetime(ltc4['time'],unit='s')
ltc4.set_index('time',inplace=True)

# combining dataframe
com1 = ltc2.append(ltc1)
com2 = ltc3.append(com1)
ltc = ltc4.append(com2)

# saving ETH data set
#ltc.to_csv("litecoin.csv")

# --Data Selection
from pandas import DataFrame, concat

df = DataFrame({'ETH': eth.close})
dataframe = concat([btc, df], axis=1)
dataframe.drop(columns = ['conversionType','conversionSymbol'], axis=1, inplace=True)

values = DataFrame(btc.close.values)
lags = 8
columns = [values]
for i in range(1,(lags + 1)):
    columns.append(values.shift(i))
dt = concat(columns, axis=1)
columns = ['Lag']
for i in range(1,(lags + 1)):
    columns.append('Lag' + str(i))
dt.columns = columns
dt.index = dataframe.index

dataframe = concat([dataframe, dt], axis=1)
#dataframe = pd.concat([dataframe, dt], axis=1)
dataframe.dropna(inplace=True)
#dataframe['day_of_month'] = dataframe.index.day
#dataframe['day_of_year'] = dataframe.index.dayofyear
dataframe['S_10'] = dataframe['close'].rolling(window=10).mean()
dataframe['Corr'] = dataframe['close'].rolling(window=10).corr(dataframe['S_10'])
dataframe['d_20'] = dataframe['close'].shift(480)
dataframe['5EMA'] = (dataframe['close'].ewm(span=5,adjust=True,ignore_na=True).mean())
dataframe['10EMA'] = (dataframe['close'].ewm(span=10,adjust=True,ignore_na=True).mean())
dataframe['20EMA'] = (dataframe['close'].ewm(span=20,adjust=True,ignore_na=True).mean())
#dataframe['48EMA'] = (dataframe['close'].ewm(span=48,adjust=True,ignore_na=True).mean())
#dataframe['96EMA'] = (dataframe['close'].ewm(span=96,adjust=True,ignore_na=True).mean())
dataframe['mean'] = (dataframe['low'] + dataframe['high'])/2
dataframe['returns'] = (dataframe['close'] - dataframe['open']) / dataframe['open'] * 100.0
dataframe['volume'] = dataframe['volumeto'] - dataframe['volumefrom']
dataframe.drop(['volumefrom', 'volumeto'], 1, inplace=True)
dataframe.dropna(inplace=True)
#data = dataframe.copy()

dataframe = dataframe.drop(['Lag'], axis=1)
dataframe = dataframe.astype(float)
#dataframe.drop(['day_of_month','day_of_year'], 1, inplace=True)
dataframe = dataframe.sort_index(ascending=True)
#dataframe.head(2)
fcast_col = 'close' # creating label
fcast_out = int(24) # prediction for next 24 hrs
dataframe['label'] = dataframe[fcast_col].shift(-fcast_out)

#dataframe = dataframe.drop(['close', 't','open', 'high', 'low', 
                            #'volumefrom', 'volumeto'], axis=1)
#dataframe = dataframe.astype(float)

# save data
dataframe.to_csv('btc_hr.csv', header=True)
#dataframe.dropna(inplace=True)

X = np.array(dataframe.drop(['label'], axis=1))

from sklearn import preprocessing
X = preprocessing.scale(X)
X_fcast_out = X[-fcast_out:]
X = X[:-fcast_out]
dataframe.dropna(inplace=True)
y = np.array(dataframe['label'])

from sklearn.model_selection import TimeSeriesSplit
from tscv import GapKFold
#gkcv = GapKFold(n_splits=5, gap_before=2, gap_after=1);
# Split the data into train and test data set
tscv = TimeSeriesSplit(n_splits=5);
for train_index, test_index in tscv.split(X, y):
    X_train, X_test = X[train_index], X[test_index];
    y_train, y_test = y[train_index], y[test_index];
    
# regression model
#from sklearn.svm import SVR    
#lm = SVR(gamma=0.005, kernel="rbf", C=50, epsilon=0.01, degree=3, max_iter=-1, 
#        shrinking=True, tol =0.001).fit(X_train, y_train)
#print(lm)

# regression model
lm = ElasticNet(alpha = 0.0001, l1_ratio = 0.5, random_state = 0).fit(X_train, y_train)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lm, X_train, y_train, cv=tscv)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

import sklearn.externals
import joblib

# Save model to file in the current working directory
joblib_file = "joblib_linmodel.pkl"  
joblib.dump(lm, joblib_file)

# Load from file
joblib_linmodel = joblib.load(joblib_file)

# prediction on training
tr_pred = joblib_linmodel.predict(X_train)
r_squared = r2_score(y_train, tr_pred)
mae = np.mean(abs(tr_pred - y_train))
rmse = np.sqrt(np.mean((tr_pred - y_train)**2))
rae = np.mean(abs(tr_pred - y_train)) / np.mean(abs(y_train - np.mean(y_train)))
rse = np.mean((tr_pred - y_train)**2) / np.mean((y_train - np.mean(y_train))**2)
sum_df = DataFrame(index = ['R-squared', 'Mean Absolute Error', 'Root Mean Squared Error',
                                'Relative Absolute Error', 'Relative Squared Error'])
sum_df['Training metrics'] = [r_squared, mae, rmse, rae, rse]

# prediction of test
te_pred = joblib_linmodel.predict(X_test)
r_squared = r2_score(y_test, te_pred)
mae = np.mean(abs(te_pred - y_test))
rmse = np.sqrt(np.mean((te_pred - y_test)**2))
rae = np.mean(abs(te_pred - y_test)) / np.mean(abs(y_test - np.mean(y_test)))
rse = np.mean((te_pred - y_test)**2) / np.mean((y_test - np.mean(y_test))**2)

sum_df['Validation metrics'] = [r_squared, mae, rmse, rae, rse]
sum_df= sum_df.round(decimals=3)
print(sum_df)

# calculate residuals
residuals = [y_test[i]-te_pred[i] for i in range(len(te_pred))]
residuals = DataFrame(residuals)
error = residuals.mean()

from sklearn.metrics import mean_absolute_error

tr_mae = mean_absolute_error(y_train, tr_pred)
te_mae = mean_absolute_error(y_test, te_pred)

# actual vs prediction test

y_test = DataFrame(y_test) # actual
y_test.index = btc.index[-len(y_test):]
y_test.rename(columns = {0: 'Actual'}, inplace = True)
#y_test.tail()

from pandas import DataFrame, concat

# Actual vs prediction validation
predict = DataFrame(te_pred) # prediction
predict.rename(columns = {0: 'Predicted'}, inplace = True)
predict.index = y_test.index


fig = go.Figure()
fig.add_trace(go.Scatter(x = btc['close'].index, y = btc['close'],
                         marker = dict(color = "red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = predict.index, y = predict['Predicted'], marker = dict(
        color = "green"), name = "Prediction"))
fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title= "Actual vs Prediction", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000)
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()

# forecast future values
fcast1 = DataFrame(joblib_linmodel.predict(X_fcast_out)); #set that will contain the forecasted data
#fcast1 = fcast1 + error # adding absolute error
fcast1 = fcast1 + error # adding absolute error

# assigning names to columns
fcast1.rename(columns = {0: 'Forecast'}, inplace = True)
d = btc.tail(fcast_out)
d.reset_index(inplace = True)
d = d.append(DataFrame({'time': pd.date_range(start = d.time.iloc[-1], 
                                             periods = (len(d)+1), freq = 'H', closed = 'right')}))
d.set_index('time', inplace = True)
d = d.tail(fcast_out)
fcast1.index = d.index
print('24 hours forecast (hourly):')
fcast1.reset_index(inplace=True)
print(fcast1)

fig = go.Figure()
fig.add_trace(go.Scatter(x = fcast1.index, y = fcast1.Forecast,
                         marker = dict(color ="teal"), name = "Forecasted price"))
fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title = "24 hrs Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000)
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()


fig = go.Figure()
n = fcast1.time[0]
fig.add_trace(go.Scatter(x = btc.index[-200:], y = btc.close[-200:],
                         marker = dict(color = "red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = fcast1['time'], y = fcast1['Forecast'], marker = dict(
        color = "green"), name = "Future prediction"))
fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title= "24 hrs close Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000,
    shapes = [dict(x0 = n, x1 = n, y0 = 0, y1 = 1, xref = 'x', yref = 'paper', line_width = 2)],
    annotations = [dict(x = n, y = 0.05, xref = 'x', yref = 'paper', showarrow = False, 
                        xanchor = 'left', text = 'Prediction')])
fig.update_layout(autosize = False, width = 1000, height = 400)
fig.show()

elapse = time() - start
print('Time elapsed:, ', elapse)


print('First program over.....')
print('*********************************************************************************')

# ARIMA model
print('Starting 2nd program (hourly data).....')

# parameters
fcast_out = int(24); mlags = int(36); order = (4,1,2)

# Future prediction
from statsmodels.tsa.arima_model import ARIMA

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

""" 
Inverting by adding the value of the observation one day ago (60*24 mins)
This is require for forecasts made by model trained on seasonally adjusted data
"""
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# seasonal difference
X = btc['close'].astype(float);
duration = int(168);
differenced = difference(X, duration);

# fit model
model = ARIMA(differenced, order = order).fit(disp=0);

# multi-step out-of-sample forecast
fcast = model.forecast(steps = fcast_out)[0];

# Walk forward validation
predict = [x for x in X];
hour = 1;

# invert the differenced forecast 
for yhat in fcast:
    inverted = inverse_difference(predict, yhat, duration);
    #print('Minute %d: %f' % (minute, inverted))
    predict.append(inverted);
    hour += 1;
        
from pandas import DataFrame
fcast2 = DataFrame(predict[-fcast_out:])
# assigning names to columns
fcast2.rename(columns = {0: 'Forecast'}, inplace=True)
d = btc.tail(fcast_out)
d.reset_index(inplace = True)
d = d.append(DataFrame({'time': pd.date_range(start = d.time.iloc[-1], 
                                             periods = (len(d)+1), freq = 'h', closed = 'right')}))
d.set_index('time', inplace=True)
d = d.tail(fcast_out)
fcast2.index = d.index
fcast2.reset_index(inplace=True)
print('Forecast for next 60 minutes:')
print(fcast2)


fig = go.Figure()
n = fcast2.time[0]
fig.add_trace(go.Scatter(x = btc.index[-200:], y = btc.close[-200:],
                         marker = dict(color ="red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = fcast2['time'], y = fcast2['Forecast'], marker=dict(
        color = "green"), name = "Future prediction"))

fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title = "24 hours Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000,
    shapes = [dict(x0 = n, x1 = n, y0 = 0, y1 = 1, xref = 'x', yref = 'paper', line_width = 2)],
    annotations = [dict(x = n, y = 0.05, xref = 'x', yref = 'paper', showarrow = False, 
                        xanchor = 'left', text = 'Prediction')]) 
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()

print('.....2nd program over....')

elapse = time() - start
print('time elapsed:,', elapse)


print('*********************************************************************************')

import threading

def tick():
    threading.Timer(3600.0, tick).start() # called every hour
    print("tick tock tick tock......see you in an hour again !!!")

tick()

print('*********************************************************************************')

print('Starting 3rd program (minute data)........')
start = time()

apiKey = "43b01c420b66888ce4c91b364647600814578c186e8604322152f44c641ebbc1"
url = "https://min-api.cryptocompare.com/data/histominute"

# BTC data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()

btc = DataFrame(result['Data'])
btc['time'] = pd.to_datetime(btc['time'],unit='s')
btc.set_index('time',inplace=True)

# parameters
fcast_out = int(60); mlags = int(60); method = 'mle'; trend = 'nc'; order = (4,1,2)

# Future prediction
from statsmodels.tsa.arima_model import ARIMA

# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return np.array(diff)

""" 
Inverting by adding the value of the observation one day ago (60*24 mins)
This is require for forecasts made by model trained on seasonally adjusted data
"""
 
# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]
 
# seasonal difference
X = btc['close'].astype(float);
duration = int(60*24);
differenced = difference(X, duration);

# fit model
model = ARIMA(differenced, order = order).fit(trend = trend, method = method, max_lag = mlags, disp=0);

# multi-step out-of-sample forecast
fcast = model.forecast(steps = fcast_out)[0];

# Walk forward validation
predict = [x for x in X];
minute = 1;

# invert the differenced forecast 
for yhat in fcast:
    inverted = inverse_difference(predict, yhat, duration);
    #print('Minute %d: %f' % (minute, inverted))
    predict.append(inverted);
    minute += 1;
    
from pandas import DataFrame
fcast3 = DataFrame(predict[-fcast_out:])
# assigning names to columns
fcast3.rename(columns = {0: 'Forecast'}, inplace=True)
d = btc.tail(fcast_out)
d.reset_index(inplace = True)
d = d.append(DataFrame({'time': pd.date_range(start = d.time.iloc[-1], 
                                             periods = (len(d)+1), freq = '1min', closed = 'right')}))
d.set_index('time', inplace=True)
d = d.tail(fcast_out)
fcast3.index = d.index
fcast3.reset_index(inplace=True)
print('Forecast for next 60 minutes:')
print(fcast3)


fig = go.Figure()
n = fcast3.time[0]
fig.add_trace(go.Scatter(x = btc.index[-500:], y = btc.close[-500:],
                         marker = dict(color ="red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = fcast3['time'], y = fcast3['Forecast'], marker=dict(
        color = "green"), name = "Future prediction"))

fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title = "60 mins close Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000,
    shapes = [dict(x0 = n, x1 = n, y0 = 0, y1 = 1, xref = 'x', yref = 'paper', line_width = 2)],
    annotations = [dict(x = n, y = 0.05, xref = 'x', yref = 'paper', showarrow = False, 
                        xanchor = 'left', text = 'Prediction')]) 
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()

elapse = time() - start
print('Time elapsed:', elapse)

import threading

def tick():
    threading.Timer(600.0, tick).start() # called every hour
    print("tick tock tick tock......see you in 10 minutes again !!!")

tick()

print('....3rd program over.....')

print('*********************************************************************************')

print('Starting 4th program (Daily data)......'); print()

start = time()

apiKey = "43b01c420b66888ce4c91b364647600814578c186e8604322152f44c641ebbc1"
url = "https://min-api.cryptocompare.com/data/histoday"

# BTC data
payload = {
    "api_key": apiKey,
    "fsym": "BTC",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()
btc = DataFrame(result['Data'])
btc['time'] = pd.to_datetime(btc['time'],unit='s')
btc.set_index('time',inplace=True)

# ETH Data
payload = {
    "api_key": apiKey,
    "fsym": "ETH",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()
eth = DataFrame(result['Data'])
eth['time'] = pd.to_datetime(eth['time'],unit='s')
eth.set_index('time',inplace=True)

# LTC data
payload = {
    "api_key": apiKey,
    "fsym": "LTC",
    "tsym": "USD",
    "limit": 2000
}

result = requests.get(url, params=payload).json()
ltc = DataFrame(result['Data'])
ltc['time'] = pd.to_datetime(ltc['time'],unit='s')
ltc.set_index('time',inplace=True)

# --Data Selection

df = DataFrame({'LTC': ltc.close,
                   'ETH': eth.close})
df = pd.concat([btc, df], axis=1)

# creating features
df['volume'] = df.volumeto - df.volumefrom
df = df.drop(columns = ['volumefrom', 'volumeto'], axis=1)

# fundamental data
#fundamental = quandl.get("BITCOINWATCH/MINING", authtoken="LSQpgUzwJRoF667ZpzyL") 
#fundamental = fundamental.fillna(0)

# adding fundamental data with dataframe
#df = pd.concat([df, fundamental], axis = 1)

# Technical Indicators

# Adding all the indicators
df = ta.add_all_ta_features(df, open = "open", high = "high", low = "low", 
                            close = "close", volume = "volume", fillna = True)

# Dropping Open', 'High', 'Low', 'Volume' 
df = df.drop(['open', 'high', 'low', 'volume'], axis=1)
#print(df.columns);print(); print(df.shape)

df = df.drop(columns = ['ETH', 'volatility_atr', 'volatility_bbhi', 'volatility_bbli',
                        'volatility_kchi','volatility_kcli', 'trend_adx', 'trend_adx_pos',
                        'trend_adx_neg', 'trend_psar_up_indicator','trend_psar_down_indicator',
                        'momentum_ao','momentum_roc'], axis=1)

# selecting 15 correlated features
X = df.drop(columns = ['close'], axis=1)
y = df['close']

# we will be keeping 15 most correlated features with 'Close' column
correlations = np.abs(X.corrwith(y))
features =  list(correlations.sort_values(ascending = False)[0:25].index)
X = X[features]
#print(X.columns); print()

"""
X columns here contain 15 selected features from above correlation
"""

#df = X.join(y, how = 'left')
df = pd.concat([X, y], axis=1)
#print(df.columns); 

fcast_col = 'close' # creating label
fcast_out = int(30) # prediction for next 10 days
#print('length =', len(df), "and forecast_out =", forecast_out); print()

df['label'] = df[fcast_col].shift(-fcast_out)
df = df.astype(float)

b = df.drop(['label', 'close'], axis=1)
# Define features Matrix X by excluding the label column which we just created 
X = np.array(b) # dropping label from feature
X_fcast_out = X[-fcast_out:]
X = X[:-fcast_out]

"""
X contains last 'n= forecast_out' rows for which we don't have label data
Put those rows in different Matrix X_forecast_out by X_forecast_out = X[end-forecast_out:end]
"""
#print ("Length of X_forecast_out:", len(X_forecast_out), "& Length of X :", len(X)); print()

y = np.array(df['label'])
y = y[:-fcast_out]
#print('Length of y: ',len(y)); print()

# Walk forward validation
# Model will see last 30 days data to predict next day's price
n_train = 365
n_records = len(X)
for i in range(n_train, n_records):
    X_train, X_test = X[0:i], X[i:i+5]
    y_train, y_test = y[0:i], y[i:i+5]
    #print('train=%d, test=%d' % (len(X_train), len(X_test)))

warnings.filterwarnings("ignore") # specify to ignore warning messages
mlags = int(5)
order = (0,1,0)
seas_ord = (0,1,2,7)
method = 'mle'
trend = 'nc'


# fit model
model4 = sm.tsa.statespace.SARIMAX(endog = y_train, exog = X_train, order = order, seasonal_order = seas_ord,
                                   enforce_invertibility = False, enforce_stationarity = False,
                                   time_varying_regression = False, mle_regression = True, maxlag = mlags,
                                   method = method).fit(trend = trend, disp = False)
ypred4 = model4.get_forecast(steps = fcast_out, exog = X_fcast_out)
fcast4 = DataFrame(ypred4.predicted_mean)

# assigning names to columns
fcast4.rename(columns = {0: 'Forecast'}, inplace=True)
d = btc.tail(fcast_out)
d.reset_index(inplace = True)
d = d.append(DataFrame({'time': pd.date_range(start = d.time.iloc[-1], 
                                             periods = (len(d)+1), freq = 'd', closed = 'right')}))
d.set_index('time', inplace = True)
d = d.tail(fcast_out)
fcast4.index = d.index
#print(fcast)

fig = go.Figure()
fig.add_trace(go.Scatter(x = fcast4.index, y = fcast4.Forecast,
                         marker = dict(color ="teal"), name = "Forecasted price"))
fig.update_xaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor = 'black', mirror = True, showspikes = True,)
fig.update_layout(
    title = "30 days Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000)
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()

# 95% prediction interval
ci4_95 = DataFrame(ypred4.conf_int(alpha = 0.05))
ci4_95.rename(columns = {0: 'lower95', 1:'upper95'}, inplace=True)
ci4_95.index = d.index
#print(ci)

# 70% prediction interval
ci4_70 = DataFrame(ypred4.conf_int(alpha = 0.30))
ci4_70.rename(columns = {0: 'lower70', 1:'upper70'}, inplace=True)
ci4_70.index = d.index

join4 = pd.concat([fcast4, ci4_95, ci4_70], axis=1)
join4 = join4.round(decimals=3)


join4.reset_index(inplace=True)
print('30 days forecast (daily)')
print(join4)

fig = go.Figure()
n = join4.time[0]
fig.add_trace(go.Scatter(x = btc.index[-200:], y = btc.close[-200:],
                         marker = dict(color ="red"), name = "Actual close price"))
fig.add_trace(go.Scatter(x = join4['time'], y = join4['Forecast'], marker=dict(
        color = "green"), name = "Future prediction"))
fig.add_trace(go.Scatter(x = join4['time'], y = join4['lower95'], name = "95% prediction interval",
                        line = dict(color ='gray', width = 4, dash = 'dot')))
fig.add_trace(go.Scatter(x = join4['time'], y = join4['upper95'], name = "95% prediction interval",
                        line=dict(color = 'gray', width = 4, dash = 'dot')))
fig.add_trace(go.Scatter(x = join4['time'], y = join4['lower70'], name = "70% prediction interval",
                        line = dict(color = 'blue', width = 4, dash = 'dot')))
fig.add_trace(go.Scatter(x = join4['time'], y = join4['upper70'], name = "70% prediction interval",
                        line = dict(color = 'blue', width = 4, dash = 'dot')))

fig.update_xaxes(showline = True, linewidth = 2, linecolor='black', mirror = True, showspikes = True,)
fig.update_yaxes(showline = True, linewidth = 2, linecolor='black', mirror = True, showspikes = True,)
fig.update_layout(
    title= "30 days close Price Forecast", 
    yaxis_title = 'BTC (US$)',
    hovermode = "x",
    hoverdistance = 100, # Distance to show hover label of data point
    spikedistance = 1000,
    shapes = [dict(x0 = n, x1 = n, y0 = 0, y1 = 1, xref = 'x', yref = 'paper',line_width = 2)],
    annotations = [dict(x = n, y = 0.05, xref = 'x', yref = 'paper', showarrow = False, xanchor = 'left', 
                        text = 'Prediction')]) 
fig.update_layout(autosize = False, width = 1000, height = 400,)
fig.show()

print('.....4th program over.....')

elapse = time() - start
print('Time elapsed:, ', elapse)

import threading

def tick():
    threading.Timer(86400.0, tick).start() # called every day
    print("tick tock tick tock......see you tomorrow again !!!")

tick()

print('*********************************************************************************')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




