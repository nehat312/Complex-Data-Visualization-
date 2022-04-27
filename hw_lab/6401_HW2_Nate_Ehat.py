#%% [markdown]
# DATS-6401 - HW #2
# Nate Ehat

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import pandas_datareader as web

import matplotlib.pyplot as plt
import seaborn as sns

# from scipy import stats as stats
# import statistics
#from tabulate import tabulate

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1

# Using the pandas_datareader package connect to yahoo database and load the stock value for the following giant companies.
# Pick the start date as ‘2000-01-01’ and the end date “ today date”.
# stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
# You will need the following package to be able to connect to yahoo API.
# Make sure to use the updated version of the pandas and pandas data_reader
# You can use the “pip install --upgrade pandas” and “pip install --upgrade pandas-datareader”


stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2000-01-01'
end_date = '2021-02-17'

print("\nVARIABLES ASSIGNED")

# Pull ticker data
aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)

stock_pulls = [aapl, orcl, tsla, ibm, yelp, msft]

print("\nSTOCKS PULLED")


#%%
# QUESTION 2
# The database contains 6 features: “High”, “Low”, “Open”, “Close”, “Volume”, “Adj Close” in USD($).
# Using the matplotlib.pyplot package and subplot command:
    # plot the “High” columns for all companies in one figure
    # 3 rows and 2 columns graph.
# Make sure to add title, legend, x- label. y-label and grid to your plot.
# The plot should look like the following. Fig size = (16,8)

#%%
# HIGH
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['High'])
plt.title('AAPL HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['High'])
plt.title('ORCL HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['High'])
plt.title('TSLA HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['High'])
plt.title('IBM HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['High'])
plt.title('YELP HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['High'])
plt.title('MSFT HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 3
# Repeat question 2 for, “Low”, “Open”, “Close”, “Volume”, “Adj Close”.

# LOW
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['Low'])
plt.title('AAPL LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['Low'])
plt.title('ORCL LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['Low'])
plt.title('TSLA LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['Low'])
plt.title('IBM LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['Low'])
plt.title('YELP LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['Low'])
plt.title('MSFT LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# OPEN
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['Open'])
plt.title('AAPL OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['Open'])
plt.title('ORCL OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['Open'])
plt.title('TSLA OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['Open'])
plt.title('IBM OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['Open'])
plt.title('YELP OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['Open'])
plt.title('MSFT OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# CLOSE
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['Close'])
plt.title('AAPL CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['Close'])
plt.title('ORCL CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['Close'])
plt.title('TSLA CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['Close'])
plt.title('IBM CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['Close'])
plt.title('YELP CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['Close'])
plt.title('MSFT CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()


#%%
# VOLUME
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['Volume'])
plt.title('AAPL VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['Volume'])
plt.title('ORCL VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['Volume'])
plt.title('TSLA VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['Volume'])
plt.title('IBM VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['Volume'])
plt.title('YELP VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['Volume'])
plt.title('MSFT VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# Adj Close
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(aapl['Adj Close'])
plt.title('AAPL ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 2)
plt.plot(orcl['Adj Close'])
plt.title('ORCL ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['Adj Close'])
plt.title('TSLA ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['Adj Close'])
plt.title('IBM ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['Adj Close'])
plt.title('YELP ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['Adj Close'])
plt.title('MSFT ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 4

# Using the matplotlib.pyplot package and hist command:
# plot the histogram plot of the “High” columns for all companies i n a 3x2 graph.
# Make sure to add title, legend, x-label. y-label and grid to your plot.
# The final plot should look like the following. # of bins = 50

# HIGH
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['High'], bins=50)
plt.title('AAPL HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['High'], bins=50)
plt.title('ORCL HIGH', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['High'], bins=50)
plt.title('TSLA HIGH', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['High'], bins=50)
plt.title('IBM HIGH', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['High'], bins=50)
plt.title('YELP HIGH', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['High'], bins=50)
plt.title('MSFT HIGH', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 5
# Repeat question 4 for, “Low”, “Open”, “Close”, “Volume”, “Adj Close”.

# LOW
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['Low'], bins=50)
plt.title('AAPL LOW', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['Low'], bins=50)
plt.title('ORCL LOW', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['Low'], bins=50)
plt.title('TSLA LOW', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['Low'], bins=50)
plt.title('IBM LOW', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['Low'], bins=50)
plt.title('YELP LOW', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['Low'], bins=50)
plt.title('MSFT LOW', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# OPEN
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['Open'], bins=50)
plt.title('AAPL OPEN', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['Open'], bins=50)
plt.title('ORCL OPEN', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['Open'], bins=50)
plt.title('TSLA OPEN', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['Open'], bins=50)
plt.title('IBM OPEN', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['Open'], bins=50)
plt.title('YELP OPEN', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['Open'], bins=50)
plt.title('MSFT OPEN', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# CLOSE
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['Close'], bins=50)
plt.title('AAPL CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['Close'], bins=50)
plt.title('ORCL CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['Close'], bins=50)
plt.title('TSLA CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['Close'], bins=50)
plt.title('IBM CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['Close'], bins=50)
plt.title('YELP CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['Close'], bins=50)
plt.title('MSFT CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# VOLUME
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['Volume'], bins=50)
plt.title('AAPL VOLUME', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['Volume'], bins=50)
plt.title('ORCL VOLUME', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['Volume'], bins=50)
plt.title('TSLA VOLUME', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['Volume'], bins=50)
plt.title('IBM VOLUME', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['Volume'], bins=50)
plt.title('YELP VOLUME', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['Volume'], bins=50)
plt.title('MSFT VOLUME', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# ADJ CLOSE
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.hist(aapl['Adj Close'], bins=50)
plt.title('AAPL ADJ CLOSE', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 2)
plt.hist(orcl['Adj Close'], bins=50)
plt.title('ORCL ADJ CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 3)
plt.hist(tsla['Adj Close'], bins=50)
plt.title('TSLA ADJ CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 4)
plt.hist(ibm['Adj Close'], bins=50)
plt.title('IBM ADJ CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 5)
plt.hist(yelp['Adj Close'], bins=50)
plt.title('YELP ADJ CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.subplot(2, 3, 6)
plt.hist(msft['Adj Close'], bins=50)
plt.title('MSFT ADJ CLOSE', fontsize=21)
plt.xlabel('SHARE PRICE ($USD)', fontsize=18)
plt.ylabel('FREQUENCY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 6
# Using the pandas package and .corr() function:
# calculate the person correlation coefficients between all 6 features for “AAPL” company.
# Display the correlation coefficient matrix through a table on the console.
# Which two feature has the highest correlation coefficient?
# Which two features has the lowest correlation coefficient?

print('AAPL CORR COEF MATRIX')
print(aapl.corr())

print('HIGHEST Correlation Coefficient: HIGH + OPEN')
print('LOWEST Correlation Coefficient: VOLUME + ADJ CLOSE')


#%%
# QUESTION 7
# Repeat question 6 for, “ORCL”, “TSLA”, “IBM”, “YELP” and “MSFT”.

print('ORCL CORR COEF MATRIX')
print(orcl.corr())
print('-------------------------------------')
print('TSLA CORR COEF MATRIX')
print(tsla.corr())
print('-------------------------------------')
print('IBM CORR COEF MATRIX')
print(ibm.corr())
print('-------------------------------------')
print('YELP CORR COEF MATRIX')
print(yelp.corr())
print('-------------------------------------')
print('MSFT CORR COEF MATRIX')
print(msft.corr())


#%%
# QUESTION 8
# Using the matplotlib.pyplot package, subplot, and scatter() function:
# plot the scatter plot for the “AAPL” company.
# You need to use the plt.subplots with 6x6 format to cover all the possible correlations between 6 feature.
# Add the calculated correlation coefficients in step 7 as a title to each subplot.
# Use two digits precision (.2f) for the correlation coefficients. Figure size = (16,16).
# The final plot should look like the following.

# HIGH
plt.figure(figsize=(16,16))
#plt.subplot(6, 6, 1)
plt.scatter(aapl['High'], aapl['High'])
plt.title({np.corrcoef(aapl['High'], aapl['High'])}, fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

#%%
plt.subplot(2, 3, 2)
plt.plot(orcl['High'])
plt.title('ORCL HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 3)
plt.plot(tsla['High'])
plt.title('TSLA HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 4)
plt.plot(ibm['High'])
plt.title('IBM HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 5)
plt.plot(yelp['High'])
plt.title('YELP HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.subplot(2, 3, 6)
plt.plot(msft['High'])
plt.title('MSFT HIGH', fontsize=21)
plt.xlabel('DATE / TIME', fontsize=18)
plt.ylabel('HIGH SHARE PRICE ($USD)', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# AAPL CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(aapl, x_vars=aapl.columns, y_vars=aapl.columns)
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 9
# Repeat question 8 for, “ORCL”, “TSLA”, “IBM”, “YELP” and “MSFT”.

#%%
# ORCL CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(orcl)
plt.title('ORCL CORRELATION PAIRPLOT')
plt.tight_layout(pad=1)
plt.show()

#%%
# TSLA CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(tsla)
plt.tight_layout(pad=1)
plt.show()

#%%
# IBM CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(ibm)
plt.tight_layout(pad=1)
plt.show()

#%%
# YELP CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(yelp)
plt.tight_layout(pad=1)
plt.show()

#%%
# MSFT CORR PLOT (PAIRPLOT)
plt.figure(figsize=(16,16))
sns.pairplot(msft)
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 10
# Alternatively, one can use Pandas package to plot the scatter matrix.
# Using pandas package plot the scatter matrix plot of the “AAPL” company
# Parameters : hist_kwds= {‘bins’ : 50} , alpha = 0.5, s = 10, diagonal = ‘kde’.
# Hint: you can use the following command : pd.plotting.scatter_matrix()
plt.figure(figsize=(16,16))
pd.plotting.scatter_matrix(aapl, hist_kwds= {'bins':50} , alpha=0.5, s=10, diagonal='kde')
plt.show()

