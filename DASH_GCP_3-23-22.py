#%% [markdown]
# DATS-6401 - CLASS 3/23/22
# Nate Ehat

#%% [markdown]
## DASH ##

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import pandas_datareader as web

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

# Assign app name
homework_ex = dash.Dash('HOMEWORK')

# Define app criteria
homework_ex.layout = html.Div([
    html.H1('HOMEWORK 1'),
    html.Button('SUBMIT', id='HW1', n_clicks=0),

    html.H1('HOMEWORK 2'),
    html.Button('SUBMIT', id='HW2', n_clicks=0),

    html.H1('HOMEWORK 3'),
    html.Button('SUBMIT', id='HW3', n_clicks=0),

    html.H1('HOMEWORK 4'),
    html.Button('SUBMIT', id='HW4', n_clicks=0),

])

homework_ex.run_server(
    port = 8100,
    host = '0.0.0.0'
    #port = 8100,
)


#%%

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Assign app name
new_app = dash.Dash('Complex Data Viz', external_stylesheets=external_stylesheets)

# Define app layout
new_app.layout = html.Div([html.H3('Complex Data Viz'),
                           dcc.Dropdown(
                               id='my-drop',
                               options=[
                                   {'label':'Introduction', 'value':'Introduction'},
                                   {'label':'Panda', 'value':'Panda'},
                                   {'label':'Seaborn', 'value':'Seaborn'},
                                   {'label':'MatPlotLib', 'value':'MatPlotLib'}
                               ], #multi=True, searchable=False, clearable=False
                           ),
                           html.Br(),
                           html.Div(id='my_output')

])

@new_app.callback(
    Output(component_id='my_output', component_property='children'),
    [Input(component_id='my_drop', component_property='value')],
)

def update_nate(input):
    return f'Selected item is {input}'

new_app.run_server(
    port = 8100,
    host = '0.0.0.0',
    #debug = True
)


#%%



#%%


#%%


#%%

