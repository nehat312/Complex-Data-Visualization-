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

#html.Div(id='slider-output-container')

# Define app layout
new_app.layout = html.Div([
    dcc.Graph(id = 'my-graph'),
    html.P('Mean'),
    dcc.Slider(id = 'mean', min = -3, max = 3, value = 0,
               marks={-3:'-3', -2:'-2', -1:'-1', 0:'0', 1:'1', 2:'2', 3:'3'}),

    html.Br(),
    html.P('std'),
    dcc.Slider(id = 'std', min = 1, max = 3, value = 1,
               marks={1:'1', 2:'2', 3:'3'}),

    html.Br(),
    html.P('Number of Samples'),
    dcc.Slider(id = 'size', min = 1, max = 10000, value = 100,
               marks={100:'100', 500:'500', 1000:'1000', 5000:'5000'}),

    html.Br(),
    html.P('Number of Bins'),
    dcc.Dropdown(id = 'bins',
                 options = [
                     {'label':20, 'value':20},
                     {'label':30, 'value':30},
                     {'label':40, 'value':40},
                     {'label':60, 'value':60},
                     {'label':80, 'value':80},
                     {'label':100, 'value':100},
                 ], value = 20
                 )

])

@new_app.callback(
    Output(component_id='my-graph', component_property='figure'),
    [Input(component_id='mean', component_property='value'),
     Input(component_id='std', component_property='value'),
     Input(component_id='bins', component_property='value'),
     Input(component_id='size', component_property='value')
     ]
)

def display_color(mean, std, bins, size):
    x = np.random.normal(mean, std, size=size)
    fig = px.histogram(x = x, nbins = bins, range_x = [-5, 5])
    return fig

new_app.run_server(
    port = 8110,
    host = '0.0.0.0',
    #debug = True
)




#%%


#%%


#%%

