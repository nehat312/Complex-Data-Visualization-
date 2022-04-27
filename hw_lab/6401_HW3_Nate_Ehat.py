#%% [markdown]
# DATS-6401 - HW #3
# 3/9/22
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

print("\nIMPORT SUCCESS")

#%% [markdown]
# PENGUINS

# * In this LAB you will practice the data visualization using seaborn package.
# * The dataset for this LAB is “ penguins”.
# * All figures in the LAB must have an appropriate title, x-label, y-label, and legend [if applicable].
# * All calculated numbers must be displayed with 2-digit precisions.

#%%
# QUESTION 1

# 1. Load the ‘penguins’ dataset from the seaborn package.
# Display the last 5 observations.
# Display the dataset statistics. Hint: describe()

penguins = sns.load_dataset('penguins')
penguin_cols = penguins.columns

print(penguins.tail())
print('*'*100)
print(penguins.describe())

#%%
# QUESTION 2
# 2. Dataset cleaning: Write a python program that check if the dataset is cleaned.
# If not, removed the missing observations from the data set.
# Display the portion of the code that perform the task here.
# Display the results that confirms the dataset is clean.

print(penguins.isnull().sum())

#%%
penguins.dropna(inplace=True)

print(penguins.isnull().sum())
print('*'*100)
print(penguins.info())

#%%
# QUESTION 3
# 3. Using the seaborn package graph the histogram plot “flipper_length_mm”.
# Use the ‘darkgrid’ style as seaborn the theme.
# Write down your observation about the graph on the console.

sns.set_theme(style='darkgrid') #'whitegrid', 'ticks', 'white', 'dark'))

plt.figure(figsize=(8,8))
sns.histplot(penguins['flipper_length_mm'], palette='mako')
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=21)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=18)
plt.ylabel(f'FREQUENCY', fontsize=18)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]

# * Flipper Length is not normally distributed across the dataset

#%%
# QUESTION 4
# 4. Change the bin width in the previous question to 3 and replot the graph. Hint: binwidth

plt.figure(figsize=(8,8))
sns.histplot(penguins['flipper_length_mm'], palette='mako', binwidth=3)
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=21)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=18)
plt.ylabel(f'FREQUENCY', fontsize=18)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 5
# 5. Change the bin numbers to 30 in the previous question and replot the graph. Hint: bins

plt.figure(figsize=(8,8))
sns.histplot(penguins['flipper_length_mm'], palette='mako', bins=30)
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'FREQUENCY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 6
# 6. Using the seaborn “displot”, graph then histogram plot per the species.
# Hint: You need to use the ‘hue’ .
# Write down your observation about the graph on the console.

plt.figure(figsize=(20,10))
sns.displot(data=penguins, y='species', hue='species', kind='hist', palette='mako')
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'SPECIES', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]

# * Flipper Length distribution is widely variable across species.
# * ADELIE species has the widest distribution range
# * CHINSTRAP species has the narrowest distribution range

#%%
# QUESTION 7
# 7. Re-graph the plot in the previous question with element=’step’.

plt.figure(figsize=(20,10))
sns.displot(data=penguins, y='species', hue='species', kind='hist', palette='mako', element='step', discrete=False, legend=True)
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'SPECIES', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 8
# 8. Using the seaborn package graph the ‘stacked’ histogram plot
# ‘flipper_lebgth_mm’ with respect to ‘species’
# Hint: multiple = ‘stack’.
# Write down your observation about the graph on the console.

sns.histplot(data=penguins,
            x = 'flipper_length_mm',
            hue='species',
            multiple='stack', #fill
            )

plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'SPECIES', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]

# * GENTOO species has the longest flipper length in mm of the three unique species
# * ADELE species has the shortest flipper length in mm of the three unique species
# * CHINSTRAP species has the widest distribution range of the three unique species

#%%
# QUESTION 9
# 9. Using the seaborn package and ‘displot’:
# Graph the histogram plot of ‘flipper_length_mm’ with respect to ‘sex’ and use the option “dodge”.
# Write down your observation about the graph on the console. Hint: multiple = ‘dodge’.

plt.figure(figsize=(10,10))
sns.histplot(data=penguins, x='flipper_length_mm', y='species', hue='sex',  multiple='dodge', legend=True)
plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'SPECIES', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%% [markdown]
# * Male penguins broadly have longer flipper length than females

#%%
# QUESTION 10
# 10. Using the seaborn package and ‘displot’:
# Graph the histogram plot of ‘flipper_lebgth_mm’ in two separate figures (not shared axis)
# Plot in one single graph (one row two columns).
# What is the most frequent range of flipper length in mm for male and female penguins?



#%% [markdown]
# Most Frequent Range of Flipper Length for each gender penguin:
# * Male:
# * Female:


#%%
# QUESTION 11
# 11. Using the seaborn package:
# Compare the distribution of ‘flipper_length_mm’ with respect to species in one graph (shared axis).
# Display graph in a normalized fashion that the bars height sum to 1.
# Hint: Use stat = ‘density’
# Which species has the larger flipper length and what is the approximate range?

plt.figure(figsize=(10,10))
sns.displot(penguins, x='flipper_length_mm', hue='species', kind='hist', stat='density')
plt.title(f'PENGUIN FLIPPER LENGTH BY SPECIES', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%% [markdown]
# * ADELIE species has the largest flipper length
# * Highest frequency of ADELIE at 180mm-190mm

#%%
# QUESTION 12
# 12. Using the seaborn package:
# Compare the distribution of ‘flipper_length_mm’ with respect to sex in one graph (shared axis)
# Display graph in a normalized fashion.
# Which sex has the larger flipper length and what is the approximate flipper length? Hint: Use stat = ‘density’

plt.figure(figsize=(10,10))
sns.displot(penguins, x='flipper_length_mm', hue='sex', kind='hist', stat='density')
plt.title(f'PENGUIN FLIPPER LENGTH BY SEX', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 13
# 13. Using the seaborn package:
# Compare the distribution of ‘flipper_length_mm’ with respect to species in one graph (shared axis)
# Display graph in a normalized fashion that the bars height sum to 1.
# Which flipper length and species is more probable ? Hint: Use stat = ‘probability’

plt.figure(figsize=(10,10))
sns.displot(penguins, x='flipper_length_mm', hue='species', kind='hist', stat='probability')
plt.title(f'PENGUIN FLIPPER LENGTH BY SPECIES', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]
# * ADELIE species has the largest flipper length
# * Highest frequency of ADELIE at 180mm-190mm is most probable

#%%
# QUESTION 14
# 14. Using the seaborn package:
# Estimate the underlying density function of flipper length
# with respect to ‘species’ and kernel density estimation.
# Plot the result. Hint: hue = ‘species’, kind = ‘kde’

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='species',
            multiple='fill',
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 15
# 15. Using the seaborn package:
# Estimate the underlying density function of flipper length with respect to ‘sex’ and kernel density estimation.
# Plot the result. Hint: hue = ‘sex’, kind = ‘kde’

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='sex',
            multiple='fill',
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH BY SEX', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 16
# 16. Repeat question 14 with argument multiple = ‘stack’

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='species',
            multiple='stack', #fill
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 17
# 17. Repeat question 15 with argument multiple = ‘stack’

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='sex',
            multiple='stack', #fill
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH BY SEX', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 18
# 18. Repeat question 14 with argument fill = True.
# Write down your observations about the graph.

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='species',
            multiple='fill',
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 19
# 19. Repeat question 15 with argument fill = True.
# Write down your observations about the graph.

sns.kdeplot(data=penguins,
            x = 'flipper_length_mm',
            bw_adjust=.2,
            cut=0,
            hue='sex',
            multiple='fill',
            legend = True
            )

plt.title(f'PENGUIN FLIPPER LENGTH BY SEX', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'DENSITY', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 20
# 20. Plot the scatter plot and the regression line in one graph:
# x-axis is ‘bill_length_mm’ and y- axis is ‘bill_depth_mm’.
# How the ‘bill_length_mm’ and ‘bill_depth_mm’ are correlated?

sns.regplot(data=penguins,
            x='bill_length_mm',
            y='bill_depth_mm',
            #kind='scatter',
            #hue='species',
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN BILL LENGTH VS DEPTH (BY SPECIES)', fontsize=16)
plt.xlabel(f'BILL LENGTH (MM)', fontsize=16)
plt.ylabel(f'BILL DEPTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 21
# 21. Using the count plot:
# Display the bar plot of the number penguins in different islands using the hue = species.
# Write down your observations about the graph?

sns.countplot(data=penguins,
              y='species',
              hue='species',
              orient='h',
              palette='mako')

plt.title(f'PENGUIN COUNT (BY SPECIES)', fontsize=16)
plt.xlabel(f'COUNT', fontsize=16)
plt.ylabel(f'SPECIES', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]
# * Far fewer CHINSTRAP species than either ADELIE or GENTOO

#%%
# QUESTION 22
# 22. Using the count plot:
# Display the bar plot of the number of male and female penguins [in the dataset] using the hue = species.
# Write down your observations about the graph?

sns.countplot(data=penguins,
              y='sex',
              hue='sex',
              orient='h',
              palette='mako')

plt.title(f'PENGUIN COUNT (BY SEX)', fontsize=16)
plt.xlabel(f'COUNT', fontsize=16)
plt.ylabel(f'SEX', fontsize=16)
plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%% [markdown]
# * Nearly equal proportion of male to female penguins

#%%
# QUESTION 23
# 23. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘bill_depth_mm’ for male and female.

sns.displot(data=penguins,
            x='bill_length_mm',
            y='bill_depth_mm',
            kind='kde',
            hue='sex',
            rug=True,
            #stat='probability',
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN BILL LENGTH VS DEPTH (BY SEX)', fontsize=16)
plt.xlabel(f'BILL LENGTH (MM)', fontsize=16)
plt.ylabel(f'BILL DEPTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 24
# 24. Plot the bivariate distribution between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
# Final plot like question 23.

sns.displot(data=penguins,
            x='bill_length_mm',
            y='flipper_length_mm',
            kind='kde',
            hue='sex',
            rug=True,
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN BILL LENGTH VS FLIPPER LENGTH (BY SEX)', fontsize=16)
plt.xlabel(f'BILL LENGTH (MM)', fontsize=16)
plt.ylabel(f'FLIPPER LENGTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 25
# 25. Plot the bivariate distribution between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female.
# Final plot like question 23.

sns.displot(data=penguins,
            x='flipper_length_mm',
            y='bill_depth_mm',
            kind='kde',
            hue='sex',
            rug=True,
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN FLIPPER LENGTH VS BILL DEPTH (BY SEX)', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'BILL DEPTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 27
# 27. Graph the bivariate distributions between “bill_length_mm” versus “bill_depth_mm” for male and female.
sns.displot(data=penguins,
            x='bill_length_mm',
            y='bill_depth_mm',
            hue='sex',
            stat='density',
            rug=True,
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN BILL LENGTH VS DEPTH (BY SEX)', fontsize=16)
plt.xlabel(f'BILL LENGTH (MM)', fontsize=16)
plt.ylabel(f'BILL DEPTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%
# QUESTION 28
# 28. Graph the bivariate distributions between ‘bill_length_mm’ versus ‘flipper_length_mm’ for male and female.
# Final plot like question 27.

sns.displot(data=penguins,
            x='bill_length_mm',
            y='flipper_length_mm',
            hue='sex',
            stat='density',
            rug=True,
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN BILL LENGTH VS FLIPPER LENGTH (BY SEX)', fontsize=16)
plt.xlabel(f'BILL LENGTH (MM)', fontsize=16)
plt.ylabel(f'FLIPPER LENGTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()

#%%
# QUESTION 29
# 29. Graph the bivariate distributions between ‘flipper_length_mm’ versus ‘bill_depth_mm’ for male and female.
# Final plot like question 27.

sns.displot(data=penguins,
            x='flipper_length_mm',
            y='bill_depth_mm',
            hue='sex',
            rug=True,
            #col='time',
            #row='smoker'
            )

plt.title(f'PENGUIN FLIPPER LENGTH VS BILL DEPTH (BY SEX)', fontsize=16)
plt.xlabel(f'FLIPPER LENGTH (MM)', fontsize=16)
plt.ylabel(f'BILL DEPTH (MM)', fontsize=16)
#plt.legend(loc='best')
plt.tight_layout(pad=1)
plt.show()


#%%

#%%