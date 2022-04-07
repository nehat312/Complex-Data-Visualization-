#%% [markdown]
# DATS-6401 - HW #4
# 4/6/22
# Nate Ehat

#%% [markdown]
## DASH ##
## GOOGLE CLOUD PLATFORM ##

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd

import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

import plotly as ply
import plotly.express as px

#import matplotlib.pyplot as plt
#import seaborn as sns

print("\nIMPORT SUCCESS")

#%%
# DEFINE STYLE SHEET
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

print("\nSUCCESS")

#%%
# DASH APP 1

# Assign app name
fresh_app = dash.Dash('HW4', external_stylesheets=external_stylesheets)

# Define app layout
fresh_app.layout = html.Div([html.H1('HOMEWORK #4', style={'textAlign':'center'}),
                             html.Br(),
                             dcc.Tabs(id='hw-questions',
                                      children=[
                                          dcc.Tab(label='Question 1', value='q1'),
                                          #dcc.Tab(label='Question 2', value='q2'),
                                          #dcc.Tab(label='Question 3', value='q3'),
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

@fresh_app.callback(
    dash.dependencies.Output(component_id='output-q1', component_property='children'),
    [dash.dependencies.Input(component_id='input1', component_property='value')]
)

def output_calc(respuesta):
    return f'THE OUTPUT VALUE IS: {respuesta}'


fresh_app.run_server(
    port = 8030,
    host = '0.0.0.0',
    #debug = True
)

#%%

# 1. Using Dash write a program that creates an input field and displays the entered data as text on the line below.
    # You need to create a callback function for the exercise.
    # Then deploy the created App through GCP and provide the working world web address into your report.

# 2. Using Dash create an app that user can input the following:
    # a. Number of cycles of sinusoidal.
    # b. Mean of the white noise.
    # c. Standard deviation of the white noise.
    # d. Number of samples.

# Then generates the data accordingly ( f(x) = sin(x) + noise ).
# Plot the function f(x) and the Fast Fourier Transform (FFT) of the generated data.
# The range of the x axis is -pi to pi. For tr FFT, you can use:
# from scipy.fft import fft
# Then deploy the created App through GCP and provide the working web address into your report.

from scipy.fft import fft

# 3. Using Dash create a drop-down menu with the items listed below.
# Once one of the items is selected, then a message should display:
    # selected item inside the dropdown menu is_____.
# Then deploy the created App through GCP and provide the working web address into your report.
# The default must be ‘Introduction’.
    # a. Introduction
    # b. Panda package
    # c. Seaborn package
    # d. Matplotlib Package
    # e. Principal Component Analysis
    # f. Outlier Detection
    # g. Interactive Visualization
    # h. Web-based App using Dash
    # i. Tableau



#%%
## CLASS REFERENCE

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
## CLASS REFERENCE
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
## CLASS REFERENCE - NORMAL DIST
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

