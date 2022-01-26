#%% [markdown]
# DATS-6401 - LAB #1
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

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
# QUESTION 2 - MANUAL
# Write a python program that calculates Pearson’s correlation coefficient
# Between two random variables x and y defined in question 1

def corr_coef_pearson(x,y):
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(xj*xj for xj in x)
  sum_y_sq = sum(yj*yj for yj in y)
  pearson_sum = sum(xj*yj for xj, yj in zip(x, y))
  numerator = pearson_sum - (sum_x * sum_y/n)
  denominator = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if denominator == 0: return 0
  return numerator / denominator

print(f'Pearson Correlation Coefficient: {corr_coef_pearson(x,y):.5f}')

# utilized stackoverflow.com for reference in coding manual Pearson function

#%%
# QUESTION 2 - AUTOMATED
# Write a python program that calculates Pearson’s correlation coefficient
# Between two random variables x and y defined in question 1

corr_coef = np.corrcoef(x, y)[0, 1]
print(corr_coef)
print(f'Pearson Correlation Coefficient: {corr_coef:.5f}')

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
print(f'Sample Pearson Correlation Coefficient Between X+Y: {corr_coef:.5f}')

#%%
# QUESTION 4
# Using the matplotlib.pyplot package in python:
    # Display the line plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.
    # Hint: You need to use plt.plot()

plt.figure(figsize=(12,12))
plt.plot(x, label='X Variables')
plt.plot(y, label='Y Variables')
plt.title(f'LINE PLOT OF X+Y:')# {r_x_y:.2f}')
plt.xlabel('SAMPLE POINTS (#)')
plt.ylabel('VALUE (#)')
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
# QUESTION 5
# Using the matplotlib.pyplot package in python:
    # Display the histogram plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.

plt.figure(figsize=(12,12))
plt.hist(x, label='X Variables', orientation='horizontal')
plt.hist(y, label='Y Variables', orientation='horizontal')
plt.title(f'HISTOGRAM PLOT OF X+Y:')
plt.xlabel('FREQUENCY (#)')
plt.ylabel('VALUE (#)')
plt.legend()
plt.grid()
plt.show()


#%%
# QUESTION 6
# Using pandas package in python read in the ‘tute1.csv’ dataset
    # Timeseries dataset with Sales, AdBudget and GDP column

df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tute1.csv', index_col=0)
print(df.head())
print(df.info())

#print(df.describe())
#col_names = df.columns
#print(col_names)

#%%
df1 = pd.to_datetime(df, infer_datetime_format=True)
print(df1.head())

#%%
# QUESTION 7
# Find the Pearson’s correlation coefficient between:
    # a. Sales & AdBudget
    # b. Sales & GDP
    # c. AdBudget & GDP

sales_ads_corr_coef = np.corrcoef(df['Sales'], df['AdBudget'])[0,1]
sales_gdp_corr_coef = np.corrcoef(df['Sales'], df['GDP'])[0,1]
ads_gdp_corr_coef = np.corrcoef(df['AdBudget'], df['GDP'])[0,1]

#%%
# QUESTION 8
# Display a message on the console that shows the following:
    # a. The sample Pearson’s correlation coefficient between Sales & AdBudget is:
    # b. The sample Pearson’s correlation coefficient between Sales & GDP is:
    # c. The sample Pearson’s correlation coefficient between AdBudget & GDP is:

print(f'Sample Pearson Correlation Coefficient between Sales & AdBudget: {sales_ads_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Sales & GDP: {sales_gdp_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between AdBudget & GDP: {ads_gdp_corr_coef:.5f}')

#%%
# QUESTION 9
# Display the line plot of Sales, AdBudget and GDP in one graph versus time
# Add an appropriate x- label, y-label, title, and legend to each graph.
# Hint: You need to us the plt.plot().

plt.figure(figsize=(12,12))
plt.plot(df['Sales'], label='Sales')
plt.plot(df['AdBudget'], label='AdBudget')
plt.plot(df['GDP'], label='GDP')
plt.title(f'LINE PLOT OF SALES / AD BUDGET / GDP:')
plt.xlabel('DATE / TIME')
plt.ylabel('VALUE ($)')
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
# QUESTION 10
# Plot the histogram plot of Sales, AdBudget and GDP in one graph.
# Add an appropriate x-label, y- label, title, and legend to each graph.
# Hint: You need to us the plt.hist().

plt.figure(figsize=(12,12))
plt.hist(df['Sales'], label='Sales', orientation='vertical')
plt.hist(df['AdBudget'], label='AdBudget', orientation='vertical')
plt.hist(df['GDP'], label='GDP', orientation='vertical')
plt.title(f'HISTOGRAM PLOT OF SALES / AD BUDGET / GDP:')
#SWITCH THESE
plt.xlabel('VALUE ($)')
plt.ylabel('FREQUENCY (#)')
plt.legend()
plt.grid()
plt.show()

#%%