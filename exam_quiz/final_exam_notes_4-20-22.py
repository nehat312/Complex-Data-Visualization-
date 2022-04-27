#%% [markdown]
# DATS-6401 - FINAL EXAM NOTES

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px

import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot
import scipy.stats as st

# import statistics
# import datetime as dt
# import pandas_datareader as web

print("\nIMPORT SUCCESS")

#%%
# DATA IMPORTS
iris = sns.load_dataset('iris')
auto = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/autos.clean.csv')
tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
diamonds = sns.load_dataset('diamonds')
penguins = sns.load_dataset('penguins')

print("\nIMPORT SUCCESS")

#%%
# LOAD DATA
df = px.data.iris()
features = df.columns.to_list()[:-2]

print(features)
print(df.info())

#%%
# SCALE DATA
X = df[features].values
X = StandardScaler().fit_transform(X)

print("\nDATA SCALED")

#%%
# PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components='mle', svd_solver='full') # 'mle'

pca.fit(X)
X_PCA = pca.transform(X)
print('ORIGINAL DIMENSIONS:', X.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
# PLOT PCA EXPLAINED VARIANCE CURVE
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

qqplot(df['traffic_volume'])
plt.title("QQ-plot of traffic volume ")
plt.show()


#%%


#%%

#14
kstest_tv = st.kstest(df['traffic_volume'],'norm')
kstest_t = st.kstest(df['temp'],'norm')
da_testtv = st.normaltest(df['traffic_volume'])
da_testt = st.normaltest(df['temp'])
shapiro_testtv = st.shapiro(df['traffic_volume'])
shapiro_testt = st.shapiro(df['temp'])

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([html.H1('test', style={'textAlign':'center'}),
                         dcc.Graph(id="my_graph"),
                         dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'dependent variable', 'value': 'var'},
                {'label': 'independent variable', 'value': 'var2'},
                    ],
            value='var'),
            html.H1(["choose option:"]),
            dcc.Dropdown(
            id='con_dropdown',
            options=[
                {'label': 'Da_k_squared', 'value': 'Da_k_squared'},
                {'label': 'K_S test', 'value': 'K_S test'},
                {'label': 'Shapiro Test', 'value': 'Shapiro Test'},

                    ],
            value='Da_k_squared',

            ),




 ])

@app.callback(
     Output(component_id='my_graph', component_property='figure'),
     [Input(component_id='dropdown', component_property='value'),
      Input(component_id='con_dropdown',component_property='value')]

)
def select_graph(value1, value2):
        if value1 == 'var':
            if value2=='K_S test':
                print(f"K-S test: statistics={kstest_tv[0]}, p-value={kstest_tv[1]}")
            elif value2=='Da_k_squared':
                print(f"da_k_squared test: statistics={da_testtv[0]:.5f}, p-value={da_testtv[1]:.5f}")
            elif value2 == 'Shapiro Test':
                print(f"Shapiro test: statistics={shapiro_testtv[0]:.5f}, p-value={shapiro_testtv[1]:.5f}")

        elif value1=='var2':
            if value2=='K_S test':
                print(f"K-S test: statistics={kstest_t[0]}, p-value={kstest_t[1]}")
            elif value2=='Da_k_squared':
                print(f"da_k_squared test: statistics={da_testt[0]:.5f}, p-value={da_testt[1]:.5f}")
            elif value2 == 'Shapiro Test':
                print(f"Shapiro test: statistics={shapiro_testt[0]:.5f}, p-value={shapiro_testt[1]:.5f}")


app.run_server(debug=True, port=3001)
