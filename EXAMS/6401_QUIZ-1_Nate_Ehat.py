#%% [markdown]
# DATS-6401 - QUIZ #1
# Nate Ehat

#%%
# LIBRARY IMPORTS

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
from scipy import stats as stats
import statistics

print("\nIMPORT SUCCESS")

#%%
# 1. Load the ‘taxis’ dataset from Seaborn package repository.
taxi = sns.load_dataset('taxis')
print(taxi.info())
print('---------------------------------------------------------------------')
print(taxi.describe())

#%%
# 2. How many samples and features are in the raw dataset?
    # Display a message on the console and fill out the blank:

taxi_rows_len = len(taxi)
taxi_cols = taxi.columns
taxi_cols_len = len(taxi_cols)

print(f'There are {taxi_rows_len} observations inside the raw dataset')
print('---------------------------------------------------------------------')
print(f'There are {taxi_cols_len} features[columns] inside the raw dataset')

#%%
# 3. Calculate the missing values [‘nan’ or ‘null’] ratio per column.
# If the ratio is more than 20% then remove the corresponding column.
# Display the following message on the console:

print(f'MISSING/NULL VALUES:')
print(taxi.isna().sum())
#print(taxi.isnull().sum())

# No columns are close to >20% missing values (maximum 45 rows of missing data any column)
# All columns are retained, as none reach 20% threshold

#%%
# Instead of dropping columns, removing observations/rows with missing values:

taxi_clean = taxi.dropna(axis=0)
taxi_clean_rows_len = len(taxi_clean)
taxi_clean_cols = taxi_clean.columns
taxi_clean_cols_len = len(taxi_clean_cols)

print(f'The cleaned dataset has {taxi_clean_rows_len} # of observations')
print('---------------------------------------------------------------------')
print(f'The cleaned dataset has {taxi_clean_cols_len} # of columns')
print('---------------------------------------------------------------------')
print(f'The list of removed columns does not exist - no columns near >20% missing values (maximum 45 rows of missing data any column)') # {taxi_cols}

#%%
# 4. Using the .isnull and .isna function make sure that there are no missing entries inside the dataset.
# If there are no missing observations:
# Dataset is cleaned; answer the rest of the questions in this quiz using the cleaned dataset.

print(f'MISSING/NULL VALUES:')
print(taxi_clean.isna().sum())
#print(taxi_clean.isnull().sum())

# No missing observations identified; moving forward with 'taxi_clean' dataset

#%%
# 5. Find the mean & variance of the ‘total’ and ‘tip’
# Display a message on the console with 2- digit precision.
# Display the information on the console

total_mean = np.mean(taxi_clean['total'])
tip_mean = np.mean(taxi_clean['tip'])
total_var = np.var(taxi_clean['total'])
tip_var = np.var(taxi_clean['tip'])

print(f'The mean of the total is ${total_mean:.2f}')
print('---------------------------------------------------------------------')
print(f'The mean of the tip is ${tip_mean:.2f}')
print('---------------------------------------------------------------------')
print(f'The variance of the total is ${total_var:.2f}')
print('---------------------------------------------------------------------')
print(f'The variance of the tip is ${tip_var:.2f}')
print('---------------------------------------------------------------------')

#%%
# 6. Create a new column inside the cleaned dataset and name it “tip_percentage”.
# Then write a program that fills out the blank and display the following on the console:

taxi_clean['tip_percentage'] = (taxi_clean['tip'] / taxi_clean['total'])
#taxi_clean.insert(loc=5, column='tip_percentage')
print(taxi_clean.head())

#%%
tip_0 = (sum(taxi_clean.tip_percentage == 0) / taxi_clean_rows_len) * 100
tip_10_15 = (sum((taxi_clean.tip_percentage >= .1) & (taxi_clean.tip_percentage < .15)) / taxi_clean_rows_len) * 100
tip_15_20 = (sum((taxi_clean.tip_percentage >= .15) & (taxi_clean.tip_percentage < .2)) / taxi_clean_rows_len) * 100
tip_20_plus = (sum(taxi_clean.tip_percentage >= .2) / taxi_clean_rows_len) * 100

print(f'{tip_0:.2f} % of passengers did not tip at all')
print(f'{tip_10_15:.2f} % of passengers tipped 10-15% of total [10 is included and 15 is excluded]')
print(f'{tip_15_20:.2f} % of passengers tipped 15-20% of total [15 is included and 20 is excluded]')
print(f'{tip_20_plus:.2f} % of passengers tipped more than 20% of total [20 is included]')
print('--------------------------------------------------------------------------------')
print(f'Majority of passengers tipped 15-20% of total.')

#%%
# 7. Plot the histogram of the “tip” and “total” in one graph
# [on top of each other, not a subplot]
# 50 as number of the bins.

# a. Add legend as “tip” and “total”.
# b. Add x-axis label as “range in USD($)”
# c. Add y-axis label as “ Frequency”.
# d. Add title as “Histogram plot of tips dataset” Add grid on.

plt.figure(figsize=(12,12))
plt.hist(taxi_clean.tip, label='TIP', bins=50, color='k')
plt.hist(taxi_clean.total, label='TOTAL', bins=50, color='g')
plt.title('HISTOGRAM PLOT OF TAXI DATASET')
plt.xlabel('RANGE IN USD($)')
plt.ylabel('FREQUENCY')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
# 8. Display the correlation coefficient matrix on the console
# Between ‘distance’, ‘fare’ and ‘tip’
# Fill out the blank and display on the console:

corr_cols = taxi_clean[['distance', 'fare', 'tip']]
print(corr_cols.corr())

#%%
print(f'The correlation coefficient between the fare & distance is {0.947958}')
print(f'The correlation coefficient between the tip & fare is {0.487101}')
print(f'The correlation coefficient between the tip & distance is {0.476980}')
print(f'The fare amount has the highest correlation coefficient with distance')
