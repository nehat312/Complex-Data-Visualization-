#%% [markdown]
# DATS-6401 - CLASS 3/23/22
# Nate Ehat

#%% [markdown]
## DASH ##

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns
#import plotly as ply
#import plotly.express as px
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
                               ], clearable=False,
                               #multi=True, searchable=False, clearable=False
                           ),

html.Br(),
html.Div(id='my_output')
])

@new_app.callback(
    Output(component_id='my_output', component_property='children'),
    [Input(component_id='my_drop', component_property='value')]
)

def update_nate(input):
    return f'Selected item is {input}'

new_app.run_server(
    port = 8000,
    host = '0.0.0.0',
    #debug = True
)


#%%

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

# Assign app name
new_app = dash.Dash('Complex Data Viz', external_stylesheets=external_stylesheets)

# Define app layout
new_app.layout = html.Div([
    dcc.Slider(
        id = 'my-slider',
        min = 0,
        max = 20,
        step = 2,
        value = 10
    ),

html.Div(id='slider-output-container')

])
@new_app.callback(
    Output(component_id='slider-output-container', component_property='children'),
    [Input(component_id='my_slider', component_property='value')]
)

def update_nate(value):
    return f'You Have Selected {value}'

new_app.run_server(
    port = 8150,
    host = '0.0.0.0',
    #debug = True
)




#%%


#%%


#%%

