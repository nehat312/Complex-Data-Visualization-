#%% [markdown]
# DATS-6401 - CLASS 3/9/22
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import pandas_datareader as web

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# from scipy import stats as stats
# import statistics
# import datetime as dt
# from statsmodels.graphics.gofplots import qqplot

print("\nIMPORT SUCCESS")

#%%

iris = sns.load_dataset('iris')
auto = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/autos.clean.csv')
#tips = sns.load_dataset('tips')
#flights = sns.load_dataset('flights')
#diamonds = sns.load_dataset('diamonds')
#penguins = sns.load_dataset('penguins')

df = px.data.iris()
features = df.columns.to_list()[:-2]


print(features)
print(auto.info())

#%%

#%%

X = df[features].values
X = StandardScaler().fit_transform(X)

#%%
pca = PCA(n_components='mle', svd_solver='full') # 'mle'

pca.fit(X)
X_PCA = pca.transform(X)
print('ORIGINAL DIMENSIONS:', X.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)

plt.show()

#%%
# AUTO DATASET

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
auto_cols = auto.select_dtypes(include=numerics)
auto_cols.info()

X = auto[auto._get_numeric_data().columns.to_list()[:-1]]
Y = auto['price']

#X.drop(columns='price', inplace=True, axis=1)

#%%
print(X)

#%%

X = StandardScaler().fit_transform(X)

#%%

# pca = PCA(n_components='mle', svd_solver='full') # 'mle'
pca = PCA(n_components=7, svd_solver='full') # 'mle'

pca.fit(X)
X_PCA = pca.transform(X)
print('ORIGINAL DIMENSIONS:', X.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')


#%%
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
#plt.grid()
plt.show()

# 7 features explain 90% of variance

#%%
# SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD]
# CONDITION NUMBER

# ORIGINAL DATA

from numpy import linalg as LA

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES {d}')
print(f'ORIGINAL DATA: CONDITIONAL NUMBER {LA.cond(X)}')


#%%
# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES {d_PCA}')
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER {LA.cond(X_PCA)}')
print('*'*58)

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET

#pca_df = pca.explained_variance_ratio_

a, b = X_PCA.shape
column = []

for i in range(b):
    column.append(f'PRINCIPAL COLUMN {i+1}')

df_PCA = pd.DataFrame(data=X_PCA, columns=column)
df_PCA = pd.concat([df_PCA, Y], axis=1)

df_PCA.info()

#%%

## DASH ##



#%%



#%%


#%%

