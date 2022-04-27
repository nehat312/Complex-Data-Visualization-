#%% [markdown]
### DATS-6401 - DATA VISUALIZATION
##### NATE EHAT

#%%
# LIBRARY IMPORTS
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics
import datetime as dt

print("\nIMPORT SUCCESS")

#%%

df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tips.csv')
df.head()

#%%
col_names = df.columns
col_names

#%%
df.describe()

#%%
total_bill_mean = np.mean(df['total_bill'])
tip_mean = np.mean(df['tip'])
total_bill_var = np.var(df['total_bill'])
tip_var = np.var(df['tip'])
total_bill_med = np.median(df['total_bill'])
tip_med = np.median(df['tip'])
total_bill_std = np.std(df['total_bill'])
tip_std = np.std(df['tip'])
total_bill_cov = np.cov(df['total_bill'])
tip_cov = np.cov(df['tip'])

print(f'Total Bill Mean: {total_bill_mean:.2f}')
print(f'Tip Mean: {tip_mean:.2f}')
print(f'Total Bill Variance: {total_bill_var:.2f}')
print(f'Tip Variance: {tip_var:.2f}')
print(f'Total Bill Median: {total_bill_med:.2f}')
print(f'Tip Median: {tip_med:.2f}')
print(f'Total Bill Std. Dev.: {total_bill_std:.2f}')
print(f'Tip Std. Dev.: {tip_std:.2f}')
print(f'Total Bill Cov.: {total_bill_cov:.2f}')
print(f'Tip Cov.: {tip_cov:.2f}')

#%%
# CORRELATION COEFFICIENT CALC
## *** NEED TO FIX THIS *** 
## SELF-CALCULATE 

n = 50
meanx = tip_mean
meany = total_bill_mean

stdx = tip_std
stdy = total_bill_std

x = np.random.normal(meanx, stdx, n)
y = np.random.normal(meany, stdy, n)

#%%

plt.figure(figsize=(12,12))
plt.hist(x)
plt.hist(y)
plt.title(f'HISTOGRAM PLOT:')# {r_x_y:.2f}')
plt.ylabel('TOTAL BILL')
plt.xlabel('TOTAL TIP')
plt.legend()
plt.grid()
plt.show()

#%%
#from toolbox import correlation_coefficient_calc

#r_x_y = correlation_coefficient_calc(tip, meal)
print(f'Correlation Coefficient - Bill vs Tip: {r_x_y:.2f}')


#%%
cov(X, Y) = (sum (x - np.mean(x)) * (y - np.mean(y)) ) * 1/(n-1)


#%%
#!pip install pandas_datareader

#%%
import pandas_datareader as web

#%%
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
df = web.DataReader('TSLA', data_source='yahoo', start='2000-01-01', end='2022-01-18')
df.describe()

#%%
stock_cols = df.columns
stock_cols

#%%

plt.figure(figsize=(12,12))
plt.plot(df['Adj Close'])
plt.title('TSLA Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Year')
#plt.xticks(df.index)
#plt.xticks(f'{range(df['Adj Close']):mm/;

#%%

#%%

#%%

# idxmax


#%%

### SEPARATE SECTION OF CLASS - TEST LAB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics
import datetime as dt

print("\nIMPORT SUCCESS")

#%%
data = np.random.randn(4,5)
print(data)


#%%

df = pd.DataFrame(data=data, columns=[['A', 'B', 'C', 'D', 'E']], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday'])
df.head()

# %%

df3 = df.copy()
for i in range(len(df)):
    df3['max'] = df.astype('float64').idxmax(axis=1)
    df3['min'] = df.astype('float64').idxmin(axis=1)
    df3.loc['max'] = df.astype('float64').idxmax(axis=0)
    df3.loc['min'] = df.astype('float64').idxmin(axis=0)

#MANUALLY REMOVE CORNER EMPTY SUBTOTALS

#%%
df3.head()

#%%


#%%
## SCRATCH - MANUAL WAY
A_max = np.max(df['A'])
B_max = np.max(df['B'])
C_max = np.max(df['C'])
D_max = np.max(df['D'])
E_max = np.max(df['E'])

# %%
