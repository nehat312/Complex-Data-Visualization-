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
# DEFINE STYLE SHEET
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

print("\nSUCCESS")

#%%
# GENERATE DASH APP
# Assign app name
fresh_app = dash.Dash('HW4', external_stylesheets=external_stylesheets)

# Define app layout
fresh_app.layout = html.Div([html.H1('HOMEWORK 4', style={'textAlign':'center'}),
                             html.Br(),
                             dcc.Tabs(id='hw-questions',
                                      children=[
                                          dcc.Tab(label='Question 1', value='q1'),
                                          dcc.Tab(label='Question 2', value='q2'),
                                          dcc.Tab(label='Question 3', value='q3'),
                                      ]),
                             html.Div(id='layout')

                             ])

q1_layout = html.Div([
    html.H1('DATAVIZ Q1'),
    html.H5('SOLUTION'),
    html.P('INPUT:'),
    dcc.Input(id='input1', type='text'),
    html.P('OUTPUT:'),
    html.Div(id='output-q1'),
])

q2_layout = html.Div([
    html.H1('DATAVIZ Q2'),
    dcc.Dropdown(id='drop-q2',
                 options=[
                     {'label':'Kyle Kuzma', 'value':'Kyle Kuzma'},
                     {'label':'LeBron James', 'value':'LeBron James'},
                 ], value='Kyle Kuzma'),

    html.Br('OUTPUT:'),
    html.Div(id='output-q2')
])

q3_layout = html.Div([
    html.H1('DATAVIZ Q3'),
    dcc.Checklist(id='checklist-q3',
                 options=[
                     {'label':'A', 'value':'A'},
                     {'label':'B', 'value':'B'},
                     {'label':'C', 'value':'C'},
                 ], value=''),

    html.Br('OUTPUT:'),
    html.Div(id='output-q3')
])

@fresh_app.callback(
    Output(component_id='layout', component_property='children'),
    [Input(component_id='hw-questions', component_property='value')]
)

def update_layout(pregunta):
    if pregunta == 'q1':
        return q1_layout
    elif pregunta == 'q2':
        return q2_layout
    elif pregunta == 'q3':
        return q3_layout

@fresh_app.callback(
    dash.dependencies.Output(component_id='output-q1', component_property='children'),
    [dash.dependencies.Input(component_id='input1', component_property='value')]
)

def output_calc(respuesta):
    return f'{respuesta}'

@fresh_app.callback(
    dash.dependencies.Output(component_id='output-q2', component_property='children'),
    [dash.dependencies.Input(component_id='drop-q2', component_property='value')]
)

def output_calc(player):
    if player == 'Kyle Kuzma':
        return f'{player} is an MVP candidate'
    elif player == 'LeBron James':
        return f'{player} needs to retire'

@fresh_app.callback(
    dash.dependencies.Output(component_id='output-q3', component_property='children'),
    [dash.dependencies.Input(component_id='checklist-q3', component_property='value')]
)

def q3_calc(choice):
    if choice == 'A':
        return f'{choice} is NOT CORRECT'
    elif choice == 'B':
        return f'{choice} is NOT CORRECT'
    elif choice == 'C':
        return f'{choice} is CORRECT'

fresh_app.run_server(
    port = 8030,
    host = '0.0.0.0',
    #debug = True
)





#%%

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

