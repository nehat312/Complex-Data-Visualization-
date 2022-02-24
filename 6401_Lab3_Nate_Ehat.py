#%% [markdown]
# DATS-6401 - LAB #3
# 2/23/22
# Nate Ehat

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import pandas_datareader as web
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='ticks') #'whitegrid', 'ticks', 'white', 'dark'

# from scipy import stats as stats
# import statistics

print("\nIMPORT SUCCESS")

#%% [markdown]
# PART 1 - COVID CASES

#%%
# QUESTION 1
# Load the dataset using pandas package.
# Clean the dataset by removing the ‘nan’ and missing data.

cases = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/CONVENIENT_global_confirmed_cases.csv')
cases.dropna(inplace=True)
case_cols = cases.columns
print(cases.head())
print(cases.info())

#%%
# Searching for China Columns
print(case_cols[57:90])

# Searching for United Kingdom Columns
print(case_cols[249:260])

#%%
# QUESTION 2
# The country “China” has multiple columns (“ China.1”, “China.2”, ...) .
# Create a new column name it “China_sum” which contains the sum of “China.1” + “China.2”, ... column wise.
# You can use the following command to perform the task:

cases['China_sum'] = cases.iloc[0:,57:90].astype(float).sum(axis=1)
# [numbers may not be accurate]

#%%
# QUESTION 3
# Repeat step 2 for the “United Kingdom”
cases['UK_sum'] = cases.iloc[0:,249:260].astype(float).sum(axis=1)

#%%

print(f'CHINA CASES:')
print(cases['China_sum'].describe())
print('---------------------------')
print(f'UNITED KINGDOME CASES:')
print(cases['UK_sum'].describe())

#%%
# QUESTION 4
# Plot the COVID confirmed cases for the following US versus the time.
# The final plot should look like below.

plt.figure(figsize=(12,8))
sns.lineplot(data=cases['US'])
plt.title('US CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)

plt.show()

#%%
# QUESTION 5
# Repeat step 4 for the “United Kingdom”, “China”, ”Germany”, ”Brazil”, “India” and “Italy”.
# The final plot should look like below.

countries = cases[['China_sum', 'UK_sum', 'US', 'Italy', 'Brazil', 'Germany', 'India']]

plt.figure(figsize=(12,8))
sns.lineplot(data=countries)
plt.title('GLOBAL CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)

plt.show()

#%%
#RESET INDEX FOR DATE
cases.set_index('Country/Region', inplace=True)
print(cases)
#%%
print(cases['US'])
#%%
# QUESTION 6
# Plot the histogram plot of the graph in Question 4.
plt.figure(figsize=(12,8))
sns.histplot(y=cases['US'], x=cases.index, bins=50)
plt.title('US CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)

plt.show()

#%%
# QUESTION 7
# Plot the histogram plot of the graph in Question 5.
# Use subplot 3x2. Not shared axis.


countries = cases[['China_sum', 'UK_sum', 'US', 'Italy', 'Brazil', 'Germany', 'India']]
plt.figure(figsize=(20,10))
plt.subplot(2, 3, 1)
sns.histplot(y=cases['China_sum'], x=cases.index, bins=50)
plt.title('CHINA CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)
plt.subplot(2, 3, 2)
sns.histplot(y=cases['UK_sum'], x=cases.index, bins=50)
plt.title('UNITED KINGDOM CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)
plt.subplot(2, 3, 3)
sns.histplot(y=cases['Italy'], x=cases.index, bins=50)
plt.title('ITALY CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)
plt.subplot(2, 3, 4)
sns.histplot(y=cases['Brazil'], x=cases.index, bins=50)
plt.title('BRAZIL CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)
plt.subplot(2, 3, 5)
sns.histplot(y=cases['India'], x=cases.index, bins=50)
plt.title('INDIA CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)
plt.subplot(2, 3, 6)
sns.histplot(y=cases['Germany'], x=cases.index, bins=50)
plt.title('GERMANY CONFIRMED COVID CASES', fontsize=21)
plt.xlabel('YEAR', fontsize=18)
plt.ylabel('CONFIRMED COVID CASES', fontsize=18)

plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 8
# Which country (from the list above) has the highest:
# mean, variance and median of # of COVID confirmed cases?

print(f'US MEAN:{np.mean(cases.US)}')
print(f'CHINA MEAN:{np.mean(cases.China_sum)}')
print(f'UK MEAN:{np.mean(cases.UK_sum)}')
print(f'ITALY MEAN:{np.mean(cases.Italy)}')
print(f'BRAZIL MEAN:{np.mean(cases.Brazil)}')
print(f'INDIA MEAN:{np.mean(cases.India)}')
print(f'GERMANY MEAN:{np.mean(cases.Germany)}')

print(f'HIGHEST MEAN: US - {np.mean(cases.US)} CASES')

#%%

print(f'US median:{np.median(cases.US)}')
print(f'CHINA median:{np.median(cases.China_sum)}')
print(f'UK median:{np.median(cases.UK_sum)}')
print(f'ITALY median:{np.median(cases.Italy)}')
print(f'BRAZIL median:{np.median(cases.Brazil)}')
print(f'INDIA median:{np.median(cases.India)}')
print(f'GERMANY median:{np.median(cases.Germany)}')

print(f'HIGHEST median: US - {np.median(cases.US)} CASES')

#%%
print(f'US variance:{np.var(cases.US):.0f}')
print(f'CHINA variance:{np.var(cases.China_sum):.0f}')
print(f'UK variance:{np.var(cases.UK_sum):.0f}')
print(f'ITALY variance:{np.var(cases.Italy):.0f}')
print(f'BRAZIL variance:{np.var(cases.Brazil):.0f}')
print(f'INDIA variance:{np.var(cases.India):.0f}')
print(f'GERMANY variance:{np.var(cases.Germany):.0f}')

print(f'HIGHEST variance: US - {np.var(cases.US):.0f} CASES')

#%% [markdown]
# PART 2 - TITANIC

#%%
# QUESTION 1
# The titanic dataset needs to be cleaned due to nan entries.
# Remove all the nan in the dataset using “”dropna()” method.
# Display the first 5 row of the dataset.


titanic = sns.load_dataset('Titanic')
titanic.dropna(inplace=True)
titanic_cols = titanic.columns
print(titanic.head())
print(titanic.info())
print(titanic_cols)

#%%
males = [titanic['sex'] == 'male']
females =  [titanic['sex'] == 'female']


#%%
# QUESTION 2
# Write a python program that plot the pie chart.
# Show the number of male and female on the titanic dataset.
# The final answer should look like bellow.

male_pct = (len(males) / len(titanic['sex'])) * 100
female_pct = (len(females) / len(titanic['sex'])) * 100


#%%
#print(male_pct)
#print(female_pct)
count=titanic['sex'].value_counts()
print(count)

males = (titanic['sex'] == 'male') / count
print(males)

#%%
#gender_labels=[male_pct, female_pct]

explode_mf = [.03, .03]

fig, ax = plt.subplots(1,1)
ax.pie(titanic.groupby('sex'))#, explode=explode_mf, autopct='%1.2f%%') #, labels=gender_labels,
plt.title('SALES COUNT PER COLOR IN %', fontsize=21)
ax.axis('square')
ax.axis('equal')
plt.show()

#%%

gender = titanic['sex'].value_counts()
colors = ['blue', 'yellow']
f, (ax1, ax2) = plt.subplots(1, 2)
ax1.pie(gender.values.tolist(),
        labels=gender.index.values.tolist(),
        colors=colors,
        startangle=90,
        autopct='%.1f%%')
ax1.set_title('MALE / FEMALE SPLITS')


colors = {'male':'blue', 'female':'yellow'}
ax2.scatter(data=titanic, x='sex', 
ax2.set_xlabel('FARE')
ax2.set_ylabel('AGE')
ax2.set_title('FARE VS AGE')

plt.show()


#%%
# QUESTION 3
# Write a python program that plot the pie chart and shows:
# percentage of male and female on the titanic dataset.
# The final answer should look like bellow.

#%%
# QUESTION 4
# Write a python program that plot the pie chart showing:
# percentage of males who survived versus the percentage of males who did not survive.
# The final answer should look like bellow.

#%%
# QUESTION 5

#%%
# QUESTION 5
# Write a python program that plot the pie chart showing:
# Percentage of females who survived versus the percentage of females who did not survive.
# The final answer should look like bellow.

#%%
# QUESTION 6
# Write a python program that plot the pie chart showing:
# Percentage passengers with first class, second class and third-class tickets.
# The final answer should look like bellow.

#%%
# QUESTION 7
# Write a python program that plot the pie chart showing:
# Survival percentage rate based on the ticket class.
# The final answer should look like bellow.

#%%
# QUESTION 8
# Write a python program that plot the pie chart showing:
# Percentage passengers who survived versus the percentage of passengers who did not survive with the first-class ticket category.
# The final answer should look like bellow.

#%%
# QUESTION 9
# Write a python program that plot the pie chart showing:
# Percentage passengers who survived versus the percentage of passengers who did not survive with the second-class ticket category.
# The final answer should look like bellow.

#%%
# QUESTION 10
# Write a python program that plot the pie chart showing:
# Percentage passengers who survived versus
# Percentage of passengers who did not survive in the third-class ticket category.


#%%
# QUESTION 11
# Using the matplotlib and plt.subplots create a dashboard
# include all the pie charts above.
# Note: Use the figure size = (16,8).
# The final answer should look like the following.


#%%
