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
# Number of samples for both x and y = 1000
# Create a random variable x: normally distributed about the mean of zero and variance 1
# Create a random variable y: normally distributed about the mean of 5 and variance of 2

n = 1000
meanx = 0
meany = 5
stdx = 1
stdy = 2

x = np.random.normal(meanx, stdx, n)
y = np.random.normal(meany, stdy, n)

#%%
# QUESTION 2
# Write a python program that calculates Pearson’s correlation coefficient
# Between two random variables x and y defined in question 1

# CORRELATION COEFFICIENT CALC
## SELF-CALCULATE - cov() function ok??

corr_coef = np.corrcoef(x, y)
print(corr_coef)
print(f'Pearson Correlation Coefficient: {corr_coef}')

#cov(X, Y) = (sum (x - np.mean(x)) * (y - np.mean(y)) ) * 1/(n-1)
#print(f'Correlation Coefficient - Bill vs Tip: {r_x_y:.2f}')

#from toolbox import correlation_coefficient_calc
#r_x_y = correlation_coefficient_calc(tip, meal)


#%%
# QUESTION 3
# Display a message on the console that shows the following information :
    # a. The sample mean of random variable x is :
    # b. The sample mean of random variable y is :
    # c. The sample variance of random variable x is :
    # d. The sample variance of random variable y is:
    # e. The sample Pearson’s correlation coefficient between x & y is:

x_mean = np.mean(x)
y_mean = np.mean(y)
x_var = np.var(x)
y_var = np.var(y)

print(f'Sample Mean of Random Variable X: {x_mean:.2f}')
print(f'Sample Mean of Random Variable Y: {y_mean:.2f}')
print(f'Sample Variance of Random Variable X: {x_var:.2f}')
print(f'Sample Variance of Random Variable Y: {y_var:.2f}')
print(f'Sample Pearson Correlation Coefficient Between X+Y: {corr_coef}')

#total_bill_med = np.median(df['total_bill'])
#tip_med = np.median(df['tip'])
#total_bill_std = np.std(df['total_bill'])
#tip_std = np.std(df['tip'])
#total_bill_cov = np.cov(df['total_bill'])
#tip_cov = np.cov(df['tip'])

#print(f'Total Bill Median: {total_bill_med:.2f}')
#print(f'Tip Median: {tip_med:.2f}')
#print(f'Total Bill Std. Dev.: {total_bill_std:.2f}')
#print(f'Tip Std. Dev.: {tip_std:.2f}')
#print(f'Total Bill Cov.: {total_bill_cov:.2f}')
#print(f'Tip Cov.: {tip_cov:.2f}')

#%%
# QUESTION 4
# Using the matplotlib.pyplot package in python:
    # Display the line plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.
    # Hint: You need to use plt.plot()

plt.figure(figsize=(12,12))
plt.plot(x)
plt.plot(y)
plt.title(f'LINE PLOT OF X+Y:')# {r_x_y:.2f}')
plt.ylabel('Y')
plt.xlabel('X')
plt.legend()
plt.grid()
plt.show()


#%%
# QUESTION 5
# Using the matplotlib.pyplot package in python:
    # Display the histogram plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.

plt.figure(figsize=(12,12))
plt.hist(x)
plt.hist(y)
plt.title(f'HISTOGRAM PLOT OF X+Y:')# {r_x_y:.2f}')
plt.ylabel('FREQUENCY')
plt.xlabel('X')
plt.legend()
plt.grid()
plt.show()


#%%
# QUESTION 6
# Using pandas package in python read in the ‘tute1.csv’ dataset
    # Timeseries dataset with Sales, AdBudget and GDP column

df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tute1.csv', index_col=0)
col_names = df.columns
print(df.head())
print(df.info())
#print(df.describe())
#print(col_names)


#%%
# QUESTION 7
# Find the Pearson’s correlation coefficient between:
    # a. Sales & AdBudget
    # b. Sales & GDP
    # c. AdBudget & GDP

sales_ads_corr_coef = np.corrcoef(df['Sales'], df['AdBudget'])
sales_gdp_corr_coef = np.corrcoef(df['Sales'], df['GDP'])
ads_gdp_corr_coef = np.corrcoef(df['AdBudget'], df['GDP'])



#%%
# QUESTION 8
# Display a message on the console that shows the following:
    # a. The sample Pearson’s correlation coefficient between Sales & AdBudget is:
    # b. The sample Pearson’s correlation coefficient between Sales & GDP is:
    # c. The sample Pearson’s correlation coefficient between AdBudget & GDP is:

print(f'Sample Pearson Correlation Coefficient between Sales & AdBudget: {sales_ads_corr_coef}')
print(f'Sample Pearson Correlation Coefficient between Sales & GDP: {sales_gdp_corr_coef}')
print(f'Sample Pearson Correlation Coefficient between AdBudget & GDP: {ads_gdp_corr_coef}')

#%%
# QUESTION 9
# Display the line plot of Sales, AdBudget and GDP in one graph versus time
# Add an appropriate x- label, y-label, title, and legend to each graph.
# Hint: You need to us the plt.plot().

plt.figure(figsize=(12,12))
plt.plot(df['Sales'])
plt.plot(df['AdBudget'])
plt.plot(df['GDP'])
plt.title(f'HISTOGRAM PLOT OF SALES / ADBUDGET / GDP:') # {r_x_y:.2f}')
plt.ylabel('FREQUENCY')
plt.xlabel('X')
plt.legend()
plt.grid()
plt.show()


#%%
# QUESTION 10
# Plot the histogram plot of Sales, AdBudget and GDP in one graph.
# Add an appropriate x-label, y- label, title, and legend to each graph.
# Hint: You need to us the plt.hist().

plt.figure(figsize=(12,12))
plt.hist(df['Sales'])
plt.hist(df['AdBudget'])
plt.hist(df['GDP'])
plt.title(f'HISTOGRAM PLOT OF SALES / ADBUDGET / GDP:') # {r_x_y:.2f}')
plt.ylabel('FREQUENCY')
plt.xlabel('X')
plt.legend()
plt.grid()
plt.show()


#%%


#%%

# idxmax table

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
## SCRATCH - MANUAL WAY
A_max = np.max(df['A'])
B_max = np.max(df['B'])
C_max = np.max(df['C'])
D_max = np.max(df['D'])
E_max = np.max(df['E'])