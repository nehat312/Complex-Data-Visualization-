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
# QUESTION 1
# Using the NumPy package in python
# Create a random variable x, normally distributed about the mean of zero and variance 1
# Create a random variable y normally distributed about the mean of 5 and variance of 2
# Number of samples for both x and y = 1000



#%%
# QUESTION 2
# Write a python program that calculates Pearson’s correlation coefficient
# Between two random variables x and y defined in question 1



#%%
# QUESTION 3
# Display a message on the console that shows the following information :
    # a. The sample mean of random variable x is :
    # b. The sample mean of random variable y is :
    # c. The sample variance of random variable x is :
    # d. The sample variance of random variable y is:
    # e. The sample Pearson’s correlation coefficient between x & y is:


#%%
# QUESTION 4
# Using the matplotlib.pyplot package in python:
    # Display the line plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.
    # Hint: You need to use plt.plot()


#%%
# QUESTION 5
# Using the matplotlib.pyplot package in python:
    # Display the histogram plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.


#%%
# QUESTION 6
# Using pandas package in python read in the ‘tute1.csv’ dataset
# Timeseries dataset with Sales, AdBudget and GDP column

#%%
# QUESTION 7
# Find the Pearson’s correlation coefficient between:
    # a. Sales & AdBudget
    # b. Sales & GDP
    # c. AdBudget & GDP


#%%
# QUESTION 8
# Display a message on the console that shows the following:
    # a. The sample Pearson’s correlation coefficient between Sales & AdBudget is:
    # b. The sample Pearson’s correlation coefficient between Sales & GDP is:
    # c. The sample Pearson’s correlation coefficient between AdBudget & GDP is:

#%%
# QUESTION 9
# Display the line plot of Sales, AdBudget and GDP in one graph versus time
# Add an appropriate x- label, y-label, title, and legend to each graph.
# Hint: You need to us the plt.plot().

#%%
# QUESTION 10
# Plot the histogram plot of Sales, AdBudget and GDP in one graph.
# Add an appropriate x-label, y- label, title, and legend to each graph.
# Hint: You need to us the plt.hist().



#%%
#!pip install pandas_datareader
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
