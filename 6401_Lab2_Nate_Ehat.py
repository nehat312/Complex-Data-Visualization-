#%% [markdown]
# DATS-6401 - LAB #2
# Nate Ehat

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web

# import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%
# DATA IMPORTS

#df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tute1.csv', index_col=0)


#%%
# QUESTION 1
# Using the pandas_datareader package connect to yahoo database
# Load the stock value for the following giant companies:
    # Stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
# You will need the following package to be able to connect to yahoo API.
# Make sure to use the updated version of the pandas and pandas’ data_reader
# Pick the start date as ‘2000-01-01’ and the end date to be Sep 8th , 2021.

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
start_date = '2000-01-01'
end_date = '2021-09-08'

aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)

#%%


#msft_cols = msft.columns
#print(msft.head())
#print(msft_cols)

#%%
plt.figure(figsize=(12,12))
plt.plot(df['Adj Close'])
plt.title('TSLA Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Year')
#plt.xticks(df.index)
#plt.xticks(f'{range(df['Adj Close']):mm/;


#%%
# idxmax

#%%
# QUESTION 2
# The database contains the stock values of 6 major giant companies.
# Each company dataset contains 6 features:
# “High”, “Low”, “Open”, “Close”, “Volume”, “Adj Close” in USD($).
# Load the data set and create a table as shown below for the mean of each attribute.
# Display the table of the console.
# There are multiple ways to create a table in python. Pick a method of your choice.

df = pd.DataFrame(data=[[msft['high'], ]],
                  columns=[['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']], index=['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT'])
print(df.head())

#%%
df3 = df.copy()
for i in range(len(df)):
    #df3['max'] = df.astype('float64').idxmax(axis=1)
    #df3['min'] = df.astype('float64').idxmin(axis=1)
    #df3.loc['max'] = df.astype('float64').idxmax(axis=0)
    #df3.loc['min'] = df.astype('float64').idxmin(axis=0)

#%%


print(df[:])

#%%
# QUESTION 3
# Repeat question 2 for the variance.


#%%
# QUESTION 4
# Repeat question 2 for the std.

#%%
# QUESTION 5
# Repeat question 2 for the median.

#%%
# QUESTION 6
# Which company has the maximum & minimum mean in each attribute?
# Add a row to the bottom of the table and the display the table on the console.


#%%
# QUESTION 7
# Which company has the maximum & minimum variance in each attribute?
# Add a row to the bottom of the table and the display the table on the console.


#%%
# QUESTION 8
# Which company has the maximum & minimum std in each attribute?
# Add a row to the bottom of the table and the display the table on the console.


#%%
# QUESTION 9
# Which company has the maximum & minimum median in each attribute?
# Add a row to the bottom of the table and the display the table on the console.


#%%
# QUESTION 10

# Calculate the correlation matrix for AAPL with all the given features.
# Display the correlation matrix on the console.
# Hint. You may use .corr() for the calculation of correlation matrix.
# Write down your observation about the correlation matrix.

#%%
# QUESTION 11
# Repeat question 10 for 'ORCL', 'TSLA', 'IBM','YELP', 'MSFT'.





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
