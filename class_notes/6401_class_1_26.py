#%% [markdown]
# DATS-6401 - CLASS 11/26
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%
np.random.seed(123)
data_1D = pd.Series([1,2,3,4,np.nan])
index = np.arange(1,7)
data_2D = pd.DataFrame(data=np.random.randn(6,4),
                       index=index, columns=list('ABCD'))
print(data_1D)
print(data_2D)

#%%
data_2D.info()

#%%
df_arrest = pd.DataFrame({'Gender':['female', 'female', 'male', 'male', 'male'],
                   'Age':['25', '18', '', '52', '33'],
                   'Weight':['250', '180', np.nan, '210', '330'],
                   'Location':['CA', 'DC', 'VA', 'MA', 'VA'],
                   'Arrest Record':['No', 'Yes', 'Yes', 'No', np.nan]
                   })

print(df_arrest.head())

#%%
df = sns.load_dataset('car_crashes')
print(df.head())

#name = sns.get_dataset_names()
#print(name)

#%%
# z-score

df2 = df[['speeding', 'alcohol', 'ins_premium', 'total']]
df2_z_score = (df2-df2.mean()) / df2.std()

plt.figure()
df2_z_score[30:].plot()
plt.show()


#%%
col_names = df.columns
total = df(col_names)[0].values

plt.figure(figsize=(12,12))
df.plot[:10](y='total')


#%%
total_speeding_corr_coef = np.corrcoef(df['total'], df['speeding'])[0,1]
total_alcohol_corr_coef = np.corrcoef(df['total'], df['alcohol'])[0,1]
total_prem_corr_coef = np.corrcoef(df['total'], df['ins_premium'])[0,1]

print(f'Sample Pearson Correlation Coefficient between Total & Speed: {total_speeding_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Total & Alcohol: {total_alcohol_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Total & Ins. Prem.: {total_prem_corr_coef:.5f}')


#%%
total_crash = df.total.values
ins_prem = df.ins_premium.values
alcohol = df.alcohol.values
speeding = df.speeding.values


#%%

plt.figure(figsize=(12,12))
plt.scatter(total_crash, alcohol)
plt.title(f'CORR COEF')
plt.xlabel('SPEED')
plt.ylabel('Crashes')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
plt.figure(figsize=(12,12))
plt.scatter(total_crash, ins_prem)
plt.title(f'CORR COEF')
plt.xlabel('SPEED')
plt.ylabel('Crashes')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
# TITANIC DATA SET
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.info())

#%%
survived_status = titanic.iloc[:,0]
print(survived_status)
#%%
print(titanic.info())

#%%
titanic_males = titanic[titanic['sex'] == 'male']
titanic_females = titanic[titanic['sex'] == 'female']
titanic_females_survived = titanic[(titanic['survived'] == 1) & (titanic['sex'] == 'female')]

titanic_50_plus = titanic[titanic['age'] >= 50]
titanic_49_under = titanic[titanic['age'] <= 49]

print(titanic_males.info()) # 577 males
print(titanic_females.info()) # 317 males
print(titanic_50_plus.info()) # 74 50 or older
print(titanic_49_under.info()) # 640 49 or under
print(titanic_females_survived.info()) # 640 49 or under

#%%
df.dropna(how='any', inplace=True)
#df.fillna()
#

#%%
titanic['age_status'] = np.where(titanic['age'] < 18, 'teen', 'adult')

print(titanic.head())

#df.rename(columns=titanic['age_status'])

#%%

titanic.drop('parch', axis=1, inplace=True)
print(titanic.columns)

#%%

titanic['age_over_mean'] = titanic['age'] / np.mean(titanic['age'])
titanic.insert(loc=4, column='age_over_mean')
print(titanic.head())

#%%
# DIAMONDS DATA SET
diamonds = sns.load_dataset('diamonds')
print(diamonds.head())
print(diamonds.info())

print(diamonds['clarity'].unique())
print(diamonds['color'].unique())

#%%
print(diamonds.pop('carat'))

#%%
index = np.arange(10, len(df))
df1 = diamonds.drop(index)

#%%
