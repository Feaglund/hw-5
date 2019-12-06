import os
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import Holt 
from statsmodels.tsa.stattools import adfuller

#2220515 GÖKHAN ERDEN


#I changed the name of csv file for Madrid's data to MADRID.csv 
#I want to take just two columns
filename = os.getcwd() + "\\MADRID.csv"
df_MADRID = pd.read_csv(filename, usecols=['Mean TemperatureC' , 'CET'])
df_MADRID['CET'] = pd.to_datetime(df_MADRID['CET'])
df_MADRID = df_MADRID.set_index('CET')
df_MADRIDnew = df_MADRID.rename(columns={'Mean TemperatureC': 'temp-madrid'})

#I need monthly average and I used 'mmavg'
mmavg = df_MADRIDnew.resample('M').mean()
mmavg.dropna(inplace=True)
# I need to drop NA values


filename = os.getcwd() + "\\brazil.csv"
df_brazil = pd.read_csv(filename, usecols=['temp' , 'date'])
df_brazil['date'] = pd.to_datetime(df_brazil['date'])
df_brazil = df_brazil.set_index('date')
df_brazilnew = df_brazil.rename(columns={'temp': 'temp-brazil'})
df_brazilnewd = df_brazilnew.loc[~(df_brazilnew==0).all(axis=1)]

# We need to read just a part of this file because it is too big file for reading
# We did this part in previous homework
# Same processes for brazil file

bmavg = df_brazilnewd.resample('M').mean()

bmavg.dropna(inplace=True)

#Now we need to decompose 
def decomp(frame,name,f,mod='Additive'):
    #frame['Date'] = pd.to_datetime(frame['Date'])
    series = frame[name]
    array = np.asarray(series, dtype=float)
    result = sm.tsa.seasonal_decompose(array,freq=f,model=mod,two_sided=False)
    result.plot()
    plt.show() 
    return result


#Test for seeing stationoraity
def test_stationarity(timeseries):
    #We need to determinde rolling statistics
    rolmean = pd.Series(timeseries).rolling(window=12).mean()
    rolstd = pd.Series(timeseries).rolling(window=12).std()
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    #Use the Dickey-Fuller test:
    print("Results of Dickey-Fuller Test:")
    array = np.asarray(timeseries, dtype='float')
    np.nan_to_num(array,copy=False)
    dftest = adfuller(array, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)
# It didn't work here in my code I couldn't find a way to solve the problem. 



