#%% [markdown]
# DATS-6401 - LAB #2
# Nate Ehat

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web
from tabulate import tabulate

# import seaborn as sns
# from scipy import stats as stats
# import statistics

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1
# Using the pandas_datareader package connect to yahoo database
# Load the stock value for the following giant companies:
    # Stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
# You will need the following package to be able to connect to yahoo API.
# Make sure to use the updated version of the pandas and pandas’ data_reader
# Pick the start date as ‘2000-01-01’ and the end date to be Sep 8th , 2021.

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2000-01-01'
end_date = '2021-09-08'

print("\nVARIABLES ASSIGNED")

#%%
#%%
# QUESTION 2
# The database contains the stock values of 6 major giant companies.
# Each company dataset contains 6 features:
# “High”, “Low”, “Open”, “Close”, “Volume”, “Adj Close” in USD($).
# Load the data set and create a table as shown below for the mean of each attribute.
# Display the table of the console.
# There are multiple ways to create a table in python. Pick a method of your choice.

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
# Generate empty DataFrames
df_mean = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_median = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_std = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_var = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])

print("\nDATAFRAMES GENERATED")

#%%
# Calculate mean values for each ticker utilizing NumPy functions
aapl_mean = ["AAPL", round(np.mean(aapl['High']), 2), round(np.mean(aapl['Low']), 2),
              round(np.mean(aapl['Open']), 2), round(np.mean(aapl['Close']), 2),
              round(np.mean(aapl['Volume']), 2), round(np.mean(aapl['Adj Close']), 2)]

orcl_mean = ["ORCL", round(np.mean(orcl['High']), 2), round(np.mean(orcl['Low']), 2),
              round(np.mean(orcl['Open']), 2), round(np.mean(orcl['Close']), 2),
              round(np.mean(orcl['Volume']), 2), round(np.mean(orcl['Adj Close']), 2)]

tsla_mean = ["TSLA", round(np.mean(tsla['High']), 2), round(np.mean(tsla['Low']), 2),
              round(np.mean(tsla['Open']), 2), round(np.mean(tsla['Close']), 2),
              round(np.mean(tsla['Volume']), 2), round(np.mean(tsla['Adj Close']), 2)]

ibm_mean = ["IBM", round(np.mean(ibm['High']), 2), round(np.mean(ibm['Low']), 2),
             round(np.mean(ibm['Open']), 2), round(np.mean(ibm['Close']), 2),
             round(np.mean(ibm['Volume']), 2), round(np.mean(ibm['Adj Close']), 2)]

yelp_mean = ["YELP", round(np.mean(yelp['High']), 2), round(np.mean(yelp['Low']), 2),
              round(np.mean(yelp['Open']), 2), round(np.mean(yelp['Close']), 2),
              round(np.mean(yelp['Volume']), 2), round(np.mean(yelp['Adj Close']), 2)]

msft_mean = ["MSFT", round(np.mean(msft['High']), 2), round(np.mean(msft['Low']), 2),
              round(np.mean(msft['Open']), 2), round(np.mean(msft['Close']), 2),
              round(np.mean(msft['Volume']), 2), round(np.mean(msft['Adj Close']), 2)]

# Calculate median values for each ticker utilizing NumPy functions
aapl_median = ["AAPL", round(np.median(aapl['High']), 2), round(np.median(aapl['Low']), 2),
              round(np.median(aapl['Open']), 2), round(np.median(aapl['Close']), 2),
              round(np.median(aapl['Volume']), 2), round(np.median(aapl['Adj Close']), 2)]

orcl_median = ["ORCL", round(np.median(orcl['High']), 2), round(np.median(orcl['Low']), 2),
              round(np.median(orcl['Open']), 2), round(np.median(orcl['Close']), 2),
              round(np.median(orcl['Volume']), 2), round(np.median(orcl['Adj Close']), 2)]

tsla_median = ["TSLA", round(np.median(tsla['High']), 2), round(np.median(tsla['Low']), 2),
              round(np.median(tsla['Open']), 2), round(np.median(tsla['Close']), 2),
              round(np.median(tsla['Volume']), 2), round(np.median(tsla['Adj Close']), 2)]

ibm_median = ["IBM", round(np.median(ibm['High']), 2), round(np.median(ibm['Low']), 2),
             round(np.median(ibm['Open']), 2), round(np.median(ibm['Close']), 2),
             round(np.median(ibm['Volume']), 2), round(np.median(ibm['Adj Close']), 2)]

yelp_median = ["YELP", round(np.median(yelp['High']), 2), round(np.median(yelp['Low']), 2),
              round(np.median(yelp['Open']), 2), round(np.median(yelp['Close']), 2),
              round(np.median(yelp['Volume']), 2), round(np.median(yelp['Adj Close']), 2)]

msft_median = ["MSFT", round(np.median(msft['High']), 2), round(np.median(msft['Low']), 2),
              round(np.median(msft['Open']), 2), round(np.median(msft['Close']), 2),
              round(np.median(msft['Volume']), 2), round(np.median(msft['Adj Close']), 2)]

# Calculate std dev values for each ticker utilizing NumPy functions
aapl_std = ["AAPL", round(np.std(aapl['High']), 2), round(np.std(aapl['Low']), 2),
              round(np.std(aapl['Open']), 2), round(np.std(aapl['Close']), 2),
              round(np.std(aapl['Volume']), 2), round(np.std(aapl['Adj Close']), 2)]

orcl_std = ["ORCL", round(np.std(orcl['High']), 2), round(np.std(orcl['Low']), 2),
              round(np.std(orcl['Open']), 2), round(np.std(orcl['Close']), 2),
              round(np.std(orcl['Volume']), 2), round(np.std(orcl['Adj Close']), 2)]

tsla_std = ["TSLA", round(np.std(tsla['High']), 2), round(np.std(tsla['Low']), 2),
              round(np.std(tsla['Open']), 2), round(np.std(tsla['Close']), 2),
              round(np.std(tsla['Volume']), 2), round(np.std(tsla['Adj Close']), 2)]

ibm_std = ["IBM", round(np.std(ibm['High']), 2), round(np.std(ibm['Low']), 2),
             round(np.std(ibm['Open']), 2), round(np.std(ibm['Close']), 2),
             round(np.std(ibm['Volume']), 2), round(np.std(ibm['Adj Close']), 2)]

yelp_std = ["YELP", round(np.std(yelp['High']), 2), round(np.std(yelp['Low']), 2),
              round(np.std(yelp['Open']), 2), round(np.std(yelp['Close']), 2),
              round(np.std(yelp['Volume']), 2), round(np.std(yelp['Adj Close']), 2)]

msft_std = ["MSFT", round(np.std(msft['High']), 2), round(np.std(msft['Low']), 2),
              round(np.std(msft['Open']), 2), round(np.std(msft['Close']), 2),
              round(np.std(msft['Volume']), 2), round(np.std(msft['Adj Close']), 2)]
#%%
# Calculate variance values for each ticker utilizing NumPy functions
aapl_var = ["AAPL", round(np.var(aapl['High']), 2), round(np.var(aapl['Low']), 2),
              round(np.var(aapl['Open']), 2), round(np.var(aapl['Close']), 2),
              round(np.var(aapl['Volume']), 2), round(np.var(aapl['Adj Close']), 2)]

orcl_var = ["ORCL", round(np.var(orcl['High']), 2), round(np.var(orcl['Low']), 2),
              round(np.var(orcl['Open']), 2), round(np.var(orcl['Close']), 2),
              round(np.var(orcl['Volume']), 2), round(np.var(orcl['Adj Close']), 2)]

tsla_var = ["TSLA", round(np.var(tsla['High']), 2), round(np.var(tsla['Low']), 2),
              round(np.var(tsla['Open']), 2), round(np.var(tsla['Close']), 2),
              round(np.var(tsla['Volume']), 2), round(np.var(tsla['Adj Close']), 2)]

ibm_var = ["IBM", round(np.var(ibm['High']), 2), round(np.var(ibm['Low']), 2),
             round(np.var(ibm['Open']), 2), round(np.var(ibm['Close']), 2),
             round(np.var(ibm['Volume']), 2), round(np.var(ibm['Adj Close']), 2)]

yelp_var = ["YELP", round(np.var(yelp['High']), 2), round(np.var(yelp['Low']), 2),
              round(np.var(yelp['Open']), 2), round(np.var(yelp['Close']), 2),
              round(np.var(yelp['Volume']), 2), round(np.var(yelp['Adj Close']), 2)]

msft_var = ["MSFT", round(np.var(msft['High']), 2), round(np.var(msft['Low']), 2),
              round(np.var(msft['Open']), 2), round(np.var(msft['Close']), 2),
              round(np.var(msft['Volume']), 2), round(np.var(msft['Adj Close']), 2)]


#%%
# Populate empty DataFrames with calculated data above
mean_dict = {'AAPL': aapl_mean, 'ORCL': orcl_mean, 'TSLA':tsla_mean, 'IBM':ibm_mean, 'YELP':yelp_mean, 'MSFT': msft_mean}
median_dict = {'AAPL': aapl_median, 'ORCL': orcl_median, 'TSLA':tsla_median, 'IBM':ibm_median, 'YELP':yelp_median, 'MSFT': msft_median}
std_dict = {'AAPL': aapl_std, 'ORCL': orcl_std, 'TSLA':tsla_std, 'IBM':ibm_std, 'YELP':yelp_std, 'MSFT': msft_std}
var_dict = {'AAPL': aapl_var, 'ORCL': orcl_var, 'TSLA':tsla_var, 'IBM':ibm_var, 'YELP':yelp_var, 'MSFT': msft_var}

for i in stocks:
    df_mean.loc[i] = mean_dict[i]
    df_median.loc[i] = median_dict[i]
    df_std.loc[i] = std_dict[i]
    df_var.loc[i] = var_dict[i]

#%%
# Add Minimum / Maximum subtotals
outputs = [df_mean, df_median, df_var, df_std]
for df in outputs:
    df.loc['MAXIMUM'] = ['MAXIMUM', np.max(df['High']), np.max(df['Low']),
                          np.max(df['Open']), np.max(df['Close']),
                          np.max(df['Volume']), np.max(df['Adj Close']),
                          ]

    df.loc['MINIMUM'] = ['MINIMUM', np.min(df['High']), np.min(df['Low']),
                          np.min(df['Open']), np.min(df['Close']),
                          np.min(df['Volume']), np.min(df['Adj Close']),
                          ]

#%%
# Reset index to display stocks / subtotals
df_mean.set_index(['Ticker'], inplace=True)
df_median.set_index(['Ticker'], inplace=True)
df_std.set_index(['Ticker'], inplace=True)
df_var.set_index(['Ticker'], inplace=True)


#%%
# GENERATE REPORT OUTPUTS
print(f'MEAN VALUES: \n{df_mean[:]}')
print(f'MEDIAN VALUES: \n{df_median[:]}')
print(f'STD DEV VALUES: \n{df_std[:]}')
print(f'VARIANCE VALUES: \n{df_var[:]}')

#%%

# QUESTION 3
# Repeat question 2 for the variance.

# Please refer to code blocks above.

# QUESTION 4
# Repeat question 2 for the std.

# Please refer to code blocks above.

# QUESTION 5
# Repeat question 2 for the median.

# Please refer to code blocks above.

#%%
# QUESTION 6
# Which company has the maximum & minimum mean in each attribute?
# Add a row to the bottom of the table and the display the table on the console.

# QUESTION 7
# Which company has the maximum & minimum variance in each attribute?
# Add a row to the bottom of the table and the display the table on the console.

# QUESTION 8
# Which company has the maximum & minimum std in each attribute?
# QUESTION 9
# Which company has the maximum & minimum median in each attribute?

# Please see function below:

# Add a row to the bottom of the table and the display the table on the console.

for df in outputs:
    #while n < len(df):
        df.loc['MAX TICKER'] = [df.index, np.max(df['High']), np.max(df['Low']),
                          np.max(df['Open']), np.max(df['Close']),
                          np.max(df['Volume']), np.max(df['Adj Close']),
                          ]
        df.loc['MIN TICKER'] = [df.index, np.min(df['High']), np.min(df['Low']),
                          np.min(df['Open']), np.min(df['Close']),
                          np.min(df['Volume']), np.min(df['Adj Close']),
                          ]

#%%
print(f'MEAN VALUES: \n{df_mean[:]}')
print(f'MEDIAN VALUES: \n{df_median[:]}')
print(f'STD DEV VALUES: \n{df_std[:]}')
print(f'VARIANCE VALUES: \n{df_var[:]}')

#%%
# QUESTION 10

# Calculate the correlation matrix for AAPL with all the given features.
# Display the correlation matrix on the console.
# Hint. You may use .corr() for the calculation of correlation matrix.
# Write down your observation about the correlation matrix.

print(aapl.corr())

# All pricing columns are very highly correlated, though volume diverges (still strong)
# This makes sense, as generally a stock trades inside a tight band of pricing within any given day.
# Whereas, the same stock may trade at elevated or depressed volumes at much greater swings in magnitude any given day.

#%%
# QUESTION 11
# Repeat question 10 for 'ORCL', 'TSLA', 'IBM','YELP', 'MSFT'.

print(orcl.corr())
print(tsla.corr())
print(ibm.corr())
print(yelp.corr())
print(msft.corr())

# similar commentary applies as in Question 10.
# same trend between pricing can be seen across additional stocks
# certain stocks swing more wildly than others in terms of volume
# would be interesting to look at some of the retail 'meme' stocks like Gamestop or AMC