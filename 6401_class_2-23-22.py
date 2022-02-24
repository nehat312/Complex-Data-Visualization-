#%% [markdown]
# DATS-6401 - CLASS 2/23/22
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import qqplot

import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%

sns.set_theme(style='darkgrid') #'whitegrid', 'ticks', 'white', 'dark'
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
diamonds = sns.load_dataset('diamonds')
penguins = sns.load_dataset('penguins')

print("\nIMPORT SUCCESS")

#%%
print(flights.describe())
print(flights.info())

#%%
sns.lineplot(data = flights,
             x = 'year',
             y = 'passengers',
             hue = 'month')
plt.show()

#%%
data = np.random.normal(size=(20,6)) + np.arange(6)/2
sns.boxplot(data=data)
plt.show()

# CATEGORIES 2+3 have outliers - remove them

#%%
sns.relplot(data=tips,
            x='total_bill',
            y='tip',
            hue='sex')

plt.show()

#%%
sns.regplot(data=tips,
            x='total_bill',
            y='tip',
            )

plt.show()

#%%
sns.boxplot(data=tips[['total_bill', 'tip']])

plt.show()
#%%
sns.relplot(data = flights,
           x = 'year',
           y = 'passengers',
           hue = 'month',
           kind='line'
           )
plt.show()

#%%
sns.relplot(data=tips,
            x='total_bill',
            y='tip',
            kind='scatter',
            hue='day',
            col='time'
            )

plt.show()

#%%
sns.relplot(data=tips,
            x='total_bill',
            y='tip',
            kind='scatter',
            hue='day',
            col='time',
            row='smoker'
            )

plt.show()

#%%

df = flights.pivot(index='month', columns='year', values='passengers')
print(df.head())
#%%
sns.heatmap(df, annot=True, fmt='d', cmap='mako', center=df.loc['Jul', 1960]) # #YlGnBu
plt.show()

#%%

sns.countplot(data=tips,
              x='day',
              order=tips['day'].value_counts(ascending=True).index)

plt.show()

#%%
sns.countplot(data=diamonds,
              y='clarity',
              order=diamonds['clarity'].value_counts(ascending=True).index)

plt.show()

#%%
sns.countplot(data=diamonds,
              y='color',
              order=diamonds['color'].value_counts(ascending=True).index,
              orient='h')

plt.show()

#%%
#sns.color_palette('tab10')
sns.color_palette('hls',8)

sns.countplot(data=diamonds,
              y='cut',
              order=diamonds['cut'].value_counts(ascending=True).index,
              orient='h',
              palette='mako')

plt.show()


#%%

#%%
sns.pairplot(data=penguins, hue='sex', palette='mako')
plt.show()

#%%

sns.kdeplot(data=tips,
            x = 'total_bill',
            bw_adjust=.2,
            cut=0,
            hue='time',
            multiple='stack',
            ) #fill
plt.show()

#%%
sns.countplot(data=tips, x='sex', hue='smoker', palette='mako')
plt.legend(loc='best')
plt.show()

#%%
diamonds.columns

#%%

sns.kdeplot(data=diamonds, x='price', log_scale=True, hue='clarity', alpha=0.5)

plt.show()

#%%

sns.kdeplot(data=tips, x='total_bill', y='tip', kind='contour')
plt.show()