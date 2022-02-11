#%% [markdown]
# DATS-6401 - HW #1
# Nate Ehat

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import pandas_datareader as web

#import matplotlib.pyplot as plt
#from tabulate import tabulate

# import seaborn as sns
# from scipy import stats as stats
# import statistics

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1
# Using the python pandas package load the titanic.csv dataset.
# Write a python program that reads all the column title and save them under variable ‘col’.
# The list of features in the dataset and the explanation is as followed:

titanic = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/titanic.csv')
col = titanic.columns
print(titanic.info())
print(col)

#%%
# QUESTION 2
# The titanic dataset needs to be cleaned due to nan entries.
# Remove all the nan in the dataset.
# Using the .describe() and .head() display the cleaned data.

titanic_drop = titanic.dropna(axis=0)
print(titanic_drop.head())
print(titanic_drop.describe())

#%%
# QUESTION 3
# The titanic dataset needs to be cleaned due to nan entries.
# Replace all the nan in the dataset with the mean of the column.
# Using the .describe() and .head() display the cleaned data.

titanic_mean = titanic.fillna(titanic.mean())
print(titanic_mean.head())
print(titanic_mean.describe())

#%%
# QUESTION 4
# The titanic dataset needs to be cleaned due to nan entries.
# Replace all the nan in the dataset with the median of the column.
# Using the .describe() and .head() display the cleaned data.

titanic_median = titanic.fillna(titanic.median())
print(titanic_median.head())
print(titanic_median.describe())

#%%
# QUESTION 5
# Using pandas in python write a program that finds the total number of passengers on board.
total_passengers = sum((titanic.Age > 0))
titanic_males = titanic[titanic['Sex'] == 'male']
titanic_females = titanic[titanic['Sex'] == 'female']

# Then finds the number of males and females on board and display the following message:
# a. The total number of passengers on board was
print(f'The total number of passengers on board was {len(titanic)}.')

# b. From the total number of passengers onboard, there were % male passengers onboard.
print(f'From the total number of passengers onboard, there were {100*(sum(titanic_males.Age>0)/len(titanic)):.2f} % male passengers onboard.') # / total_passengers

# From the total number of passengers onboard, there were % female passengers # onboard.
    # (Display the percentage value here with 2-digit precision)
print(f'From the total number of passengers onboard, there were {100*(1-(sum(titanic_males.Age>0)/len(titanic))):.2f} % female passengers onboard.')

# d. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 6
# Using pandas in python write a program that find out the number of survivals.
# Also, the total number of male survivals and total number of female survivals.
    # Note: You need to use the Boolean & condition to filter the DataFrame.

# Then display a message on the console as follows:
# a. Total number of survivals onboard was .
titanic_survived = titanic[(titanic['Survived'] == 1)]
print(f'Total number of survivals onboard was {sum(titanic_survived.Age>0):.0f}.')

# b. Out of male passengers onboard only % male was survived.
    # (Display the percentage value here with 2-digit precision)
titanic_males_survived = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'male')]
print(f'Out of male passengers onboard only {100*(sum(titanic_males_survived.Age>0)/len(titanic)):.2f} % male was survived.')

# c. Out of female passengers onboard only % female was survived.
    # (Display the percentage value here with 2-digit precision).
titanic_females_survived = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'female')]
print(f'Out of female passengers onboard only {100*(sum(titanic_females_survived.Age>0)/len(titanic)):.2f} % female was survived.')

# d. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 7
# Using pandas in python write a program that find out the number of passengers with upper class ticket.
# Then, find out the total number of male and female survivals with upper class ticket.
# Then display the following messages on the console:

# a. There were total number of passengers with the upper-class ticket and only % were survived.
    # (Display the percentage value here with 2-digit precision).

# b. Out of passengers with upper class ticket, % passengers were male.
    # (Display the percentage value here with 2-digit precision).

# c. Out of passengers with upper class ticket, % passengers were female.
    # (Display the percentage value here with 2-digit precision).

# d. Out of passengers with upper class ticket, % male passengers were survived.
    # (Display the percentage value here with 2-digit precision).
# e. Out of passengers with upper class ticket, % female passengers were survived.
    # (Display the percentage value here with 2-digit precision).
# f. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 8
# Using pandas & numpy package add a column to the dataset called “Above50&Male”.
# Entries corresponding to this column must display “Yes” if a passenger is Male & more than 50 yo, otherwise “No”.
# Display the first 5 rows of the new dataset with the added column.
# How many male passengers onboard above 50 years old? Display the information on the console.

titanic['Above50Male'] = np.where((titanic['Age'] > 50) & (titanic['Sex'] == 'male'), 'Yes', 'No')
male50plus = sum((titanic.Age > 50) & (titanic.Sex == 'male'))
print(titanic.head())
print('------------------------------------------------------')
print(f'Male Passengers Over 50 Years Old: {male50plus}')

#%%
# QUESTION 9

# Using pandas & numpy package add a column to the dataset called “Above50&Male&Survived”.
# Entries corresponding to this column must display “Yes” if a passenger is Male & more than 50 yo & survived, otherwise “No”.
# Display the first 5 rows of the new dataset with the added column.
# How many male passengers onboard were above 50 & survived? Display the information on the console.
# Find the survival percentage rate of male passengers onboard who are above 50 years old?

titanic['Above50Male_Survived'] = np.where((titanic['Age'] >= 51) & (titanic['Sex'] == 'male') & (titanic['Survived'] == 1), 'Yes', 'No')
male50plus_survived = sum((titanic.Age > 50) & (titanic.Sex == 'male') & (titanic.Survived == 1))
print(titanic.head())
print('------------------------------------------------------')
print(f'Male Survivors Over 50 Years Old: {male50plus_survived}')

#%%
# QUESTION 10
# Repeat question 8 & 9 for the female passengers.

titanic['Above50Female'] = np.where((titanic['Age'] > 50) & (titanic['Sex'] == 'female'), 'Yes', 'No')
female50plus = sum((titanic.Age > 50) & (titanic.Sex == 'female'))
print(titanic.head())
print('------------------------------------------------------')
print(f'Female Passengers Over 50 Years Old: {female50plus}')

titanic['Above50Female_Survived'] = np.where((titanic['Age'] >= 51) & (titanic['Sex'] == 'female') & (titanic['Survived'] == 1), 'Yes', 'No')
female50plus_survived = sum((titanic.Age > 50) & (titanic.Sex == 'female') & (titanic.Survived == 1))
print(titanic.head())
print('------------------------------------------------------')
print(f'Female Survivors Over 50 Years Old: {female50plus_survived}')


#%%

# SCRATCH NOTES
survived_status = titanic.iloc[:,0]
print(survived_status)

titanic['age_over_mean'] = titanic['age'] / np.mean(titanic['age'])
print(titanic.head())
#titanic.insert(loc=4, column='age_over_mean')