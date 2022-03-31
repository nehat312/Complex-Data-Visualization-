#%% [markdown]
# DATS-6401 - CLASS 3/30/22
# Nate Ehat

#%% [markdown]
## DASH ##
## GOOGLE CLOUD PLATFORM ##

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
import plotly as ply
import plotly.express as px
#import pandas_datareader as web

import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

# from scipy import stats as stats
# import statistics
# import datetime as dt
# from statsmodels.graphics.gofplots import qqplot
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

print("\nIMPORT SUCCESS")


#%%
# DEFINE STYLE SHEET
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

print("\nSUCCESS")

#%%
df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/CONVENIENT_global_confirmed_cases.csv')
df.dropna(axis=0, how='any')
df.drop(index=df.index[0],
        axis=0,
        inplace=True)
#df.head()
print(df.info())
print(df.columns)


#%%
col_name = []
df['china_sum'] = df.iloc[0:,57:90].astype(float).sum(axis=1)
df['UK_sum'] = df.iloc[0:,249:260].astype(float).sum(axis=1)
print(df.columns)

#%%
for col in df.columns:
    col_name.append(col)
df_covid = df[col_name]
df_covid['date'] = pd.date_range(start='1-23-20', end='11-22-20')

#%%

# Assign app name
covid_app = dash.Dash('GLOBAL COVID CASES', external_stylesheets=external_stylesheets)

#server = covid_app.server

# Define app layout
covid_app.layout = html.Div([
    dcc.Graph(id='covid-graph'),

    html.P('SELECT COUNTRY NAME'),
    dcc.Dropdown(id='country',
                 options=[{'label':'US', 'value':'US'},
                          {'label':'Brazil', 'value':'Brazil'},
                          {'label':'UK_sum', 'value':'UK_sum'},
                          {'label':'Germany', 'value':'Germany'},
                          {'label':'India', 'value':'India'},
                          {'label':'Italy', 'value':'Italy'},
                          ], value='US', clearable=False)
])


@covid_app.callback(
    Output(component_id='covid-graph', component_property='figure'),
    Input(component_id='country', component_property='value'),
)

def display_chart(country):
    fig = px.line(df_covid, x='date', y=[country])
    return fig

covid_app.run_server(
    port = 8035,
    host = '0.0.0.0',
)

#%%

#%%
