#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:46:57 2024

@author: allisonforsyth
"""
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import numpy as np

os.getcwd()
os.chdir('/Users/allisonforsyth/Documents/GMU_Spring/CS_504')

import pandas as pd
df = pd.read_csv("nypd.csv")

df[['m', 'd', 'y']] = df['OCCUR_DATE'].str.split('/', expand=True)
df['m'] = df['m'].str.zfill(2)
df['d'] = df['d'].str.zfill(2)
df['date']=df["m"]+df["d"]+df["y"]
df["date"]=pd.to_datetime(df['date'],format='%m%d%y')
df['yr_m'] = df['date'].dt.strftime('%Y-%m')
df = df.set_index('date')

df = df.sort_index().loc['2006-01-21' : '2021-11-15', :]


def model_eval(y, predictions):
    
    from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
    rmse = np.sqrt(mean_squared_error(y, predictions))
    mfe = np.mean(y - predictions)
    error = y - predictions
    mfe = np.sqrt(np.mean(predictions**2))
    rmse = np.sqrt(np.mean(error**2))
    
    print('Root Mean Squared Error:', round(rmse, 3))
    print('Mean forecast error:', round(mfe, 3))
    
#Queens
queens = df[df["BORO"]=="QUEENS"]
queens

#queens_month=queens.groupby(['yr_m']).size().reset_index(name='counts')


queens_month = queens['2006-01-01':'2021-11-30'].resample('M').agg({'INCIDENT_KEY':'size'})
queens_month.head()

y = queens_month['INCIDENT_KEY']

train = y[:'2020-09-30'] # dataset to train
test = y['2020-10-31':] # last X months for test  
pred = len(y) - len(y[:'2020-10-30']) # the number of data points for the test set

result = seasonal_decompose(y, model='multiplicative', period=12)
result.plot()
plt.show()

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Prediction')
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Queens Shooting Incident Forecast Model")
plt.ylim(ymin=0)

model_eval(test, pred)

residuals = pd.DataFrame(test-pred, columns=["residuals"])
plt.figure(figsize=(15,7))
res = stats.probplot(residuals["residuals"], plot=plt)
ax = plt.gca()


model = ExponentialSmoothing(y, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start="2021-12-30", end="2022-12-30")

RMSFE = np.sqrt(sum([x**2 for x in residuals["residuals"]]) / len(residuals))
band_size = 1.96*RMSFE


plt.plot(y.index, y, label='Actual')
plt.plot(pred.index, pred, label='Forecast')
plt.fill_between(pred.index, (pred-band_size) , (pred+band_size) , color='orange', alpha=.3)
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Queens Shooting Incident Forecast Model next 12 Months")
plt.ylim(ymin=0)


#manhattan
man = df[df["BORO"]=="MANHATTAN"]
man

man_month = man['2006-01-01':'2021-11-30'].resample('M').agg({'INCIDENT_KEY':'size'})
man_month.head()

y = man_month['INCIDENT_KEY']

train = y[:'2020-09-30'] # dataset to train
test = y['2020-10-31':] # last X months for test  
pred = len(y) - len(y[:'2021-07-30']) # the number of data points for the test set

result = seasonal_decompose(y, model='multiplicative', period=12)
result.plot()
plt.show()

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Manhattan Shooting Incident Forecast Model")
plt.ylim(ymin=0)

#evaluate model
model_eval(test, pred)

residuals = pd.DataFrame(test-pred, columns=["residuals"])
plt.figure(figsize=(15,7))
res = stats.probplot(residuals["residuals"], plot=plt)
ax = plt.gca()


model = ExponentialSmoothing(y, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start="2021-12-30", end="2022-12-30")

RMSFE = np.sqrt(sum([x**2 for x in residuals["residuals"]]) / len(residuals))
band_size = 1.96*RMSFE


plt.plot(y.index, y, label='Actual')
plt.plot(pred.index, pred, label='Forecast')
plt.fill_between(pred.index, (pred-band_size) , (pred+band_size) , color='orange', alpha=.3)
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Manhattan Shooting Incident Forecast Model next 12 Months")
plt.ylim(ymin=0)




#staten island
statI = df[df["BORO"]=="STATEN ISLAND"]
statI

statI_month = statI['2006-01-01':'2021-11-30'].resample('M').agg({'INCIDENT_KEY':'size'})
statI_month.head()

y = statI_month['INCIDENT_KEY']

train = y[:'2020-09-30'] # dataset to train
test = y['2020-10-31':] # last X months for test  
pred = len(y) - len(y[:'2021-07-30']) # the number of data points for the test set

result = seasonal_decompose(y, model='add', period=12)
result.plot()
plt.show()

model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Staten Island Shooting Incident Forecast Model")
plt.ylim(ymin=0)

#evaluate model
model_eval(test, pred)

residuals = pd.DataFrame(test-pred, columns=["residuals"])
plt.figure(figsize=(15,7))
res = stats.probplot(residuals["residuals"], plot=plt)
ax = plt.gca()


model = ExponentialSmoothing(y, seasonal='add', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start="2021-12-30", end="2022-12-30")

RMSFE = np.sqrt(sum([x**2 for x in residuals["residuals"]]) / len(residuals))
band_size = 1.96*RMSFE


plt.plot(y.index, y, label='Actual')
plt.plot(pred.index, pred, label='Forecast')
plt.fill_between(pred.index, (pred-band_size) , (pred+band_size) , color='orange', alpha=.3)
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Staten Island Shooting Incident Forecast Model next 12 Months")
plt.ylim(ymin=0)


#bronx
bronx = df[df["BORO"]=="BRONX"]
bronx

bronx_month = bronx['2006-01-01':'2021-11-30'].resample('M').agg({'INCIDENT_KEY':'size'})
bronx_month.head()

y = bronx_month['INCIDENT_KEY']

train = y[:'2020-09-30'] # dataset to train
test = y['2020-10-31':] # last X months for test  
pred = len(y) - len(y[:'2020-10-31']) # the number of data points for the test set

result = seasonal_decompose(y, model='multiplicative', period=12)
result.plot()
plt.show()

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Bronx Shooting Incident Forecast Model")
plt.ylim(ymin=0)

#evaluate model
model_eval(test, pred)

residuals = pd.DataFrame(test-pred, columns=["residuals"])
plt.figure(figsize=(15,7))
res = stats.probplot(residuals["residuals"], plot=plt)
ax = plt.gca()


model = ExponentialSmoothing(y, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start="2021-12-30", end="2022-12-30")

RMSFE = np.sqrt(sum([x**2 for x in residuals["residuals"]]) / len(residuals))
band_size = 1.96*RMSFE


plt.plot(y.index, y, label='Actual')
plt.plot(pred.index, pred, label='Forecast')
plt.fill_between(pred.index, (pred-band_size) , (pred+band_size) , color='orange', alpha=.3)
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Bronx Shooting Incident Forecast Model next 12 Months")
plt.ylim(ymin=0)



#brooklyn 
brooklyn = df[df["BORO"]=="BROOKLYN"]
brooklyn

brooklyn_month = brooklyn['2006-01-01':'2021-11-30'].resample('M').agg({'INCIDENT_KEY':'size'})
brooklyn_month.head()

y = brooklyn_month['INCIDENT_KEY']

train = y[:'2020-09-30'] # dataset to train
test = y['2020-10-31':] # last X months for test  
pred = len(y) - len(y[:'2019-10-31']) # the number of data points for the test set

result = seasonal_decompose(y, model='multiplicative', period=12)
result.plot()
plt.show()

model = ExponentialSmoothing(train, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start=test.index[0], end=test.index[-1])

plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test')
plt.plot(pred.index, pred, label='Holt-Winters')
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Brooklyn Shooting Incident Forecast Model")
plt.ylim(ymin=0)

#evaluate model
model_eval(test, pred)

residuals = pd.DataFrame(test-pred, columns=["residuals"])
plt.figure(figsize=(15,7))
res = stats.probplot(residuals["residuals"], plot=plt)
ax = plt.gca()


model = ExponentialSmoothing(y, seasonal='mul', seasonal_periods=12)
hw_model = model.fit(optimized=True, remove_bias=False)
pred = hw_model.predict(start="2021-12-30", end="2022-12-30")

RMSFE = np.sqrt(sum([x**2 for x in residuals["residuals"]]) / len(residuals))
band_size = 1.96*RMSFE


plt.plot(y.index, y, label='Actual')
plt.plot(pred.index, pred, label='Forecast')
plt.fill_between(pred.index, (pred-band_size) , (pred+band_size) , color='orange', alpha=.3)
plt.legend(loc='upper left')
plt.xlabel("Date")
plt.ylabel('Number of Shootings')
plt.title("Brooklyn Shooting Incident Forecast Model next 12 Months")
plt.ylim(ymin=0)






