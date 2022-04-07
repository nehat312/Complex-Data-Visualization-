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
from scipy.fft import fft
from dash.exceptions import PreventUpdate

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

# 1. Using Dash write a program that creates an input field and displays the entered data as text on the line below.
    # You need to create a callback function for the exercise.
    # Then deploy the created App through GCP and provide the working world web address into your report.

hw4_app = dash.Dash('HOMEWORK 4', external_stylesheets=external_stylesheets)

hw4_app.layout = html.Div([html.H1('HOMEWORK 4', style={'textAlign': 'center'}),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                                   children=[
                                       dcc.Tab(label='QUESTION 1', value='q1'),
                                       dcc.Tab(label='QUESTION 2', value='q2'),
                                       dcc.Tab(label='QUESTION 3', value='q3')]),
                          html.Div(id='layout')])

q1_layout = html.Div([html.H1('Question 1'),
                             html.H5('Change the value in the textbox to see callbacks in action!'),
                             html.P('Input:'),
                             dcc.Input(id='input1',type='text'),
                             html.P(id='output1')
                             ])


@hw4_app.callback(Output(component_id='output1', component_property='children'),
                 [Input(component_id='input1', component_property='value')])
def update_q1(text):
    return f'The output value is "{text}"'

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

q2_layout = html.Div([html.H1('Question 2'),
                             html.H6('Please enter the number of sinusoidal cycle'),
                             dcc.Input(id='ip1',type='number'),
                             html.H6('Please enter the mean of white noise'),
                             dcc.Input(id='ip2',type='number'),
                             html.H6('Please enter the standard deviation of the white noise'),
                             dcc.Input(id='ip3',type='number'),
                             html.H6('Please enter the number of samples'),
                             dcc.Input(id='ip4',type='number'),
                             dcc.Graph(id='gph1'),
                             html.H6('The Fast Fourier Transform of Above Generated Data'),
                             dcc.Graph(id='gph2'),
                             ])


@hw4_app.callback([Output(component_id='gph1', component_property='figure'),
                 Output(component_id='gph2', component_property='figure')],
                 [Input(component_id='ip1', component_property='value'),
                  Input(component_id='ip2', component_property='value'),
                  Input(component_id='ip3', component_property='value'),
                  Input(component_id='ip4', component_property='value')])

def update_q2(b, m, std, n):
    if (b == None) | (m == None) | (std==None) | (n==None):
        raise PreventUpdate
    else:
        x = np.linspace(-np.pi,np.pi,n)
        noise = np.random.normal(m,std,n)
        y = np.sin(b*x) + noise
        f = abs(fft(y,n))
        fig1 = px.line(x=x, y=y)
        fig2 = px.line(x=x, y=f)
        return fig1, fig2

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

q3_layout = html.Div([
    html.H1('Complex Data Visualization'),
    dcc.Dropdown(id='drop3',
                 options=[{'label': 'Introduction', 'value': 'Introduction'},
                          {'label': 'Pandas Package', 'value': 'Pandas Package'},
                          {'label': 'Seaborn Package', 'value': 'Seaborn Package'},
                          {'label': 'Matplotlib Package', 'value': 'Matplotlib Package'},
                          {'label': 'Principal Component Analysis', 'value': 'Principal Component Analysis'},
                          {'label': 'Outlier Detection', 'value': 'Outlier Detection'},
                          {'label': 'Interactive Visualization', 'value': 'Interactive Visualization'},
                          {'label': 'Web-based App using Dash', 'value': 'Web-based App using Dash'},
                          {'label': 'Tableau', 'value': 'Tableau'}],
                 value='Introduction'),
    html.Br(),
    html.Div(id='output3')
])

@hw4_app.callback(Output(component_id='output3', component_property='children'),
                 [Input(component_id='drop3', component_property='value')])
def update_q3(input):
    return f'The selected item inside the dropdown menu is {input}'

# APP CALLBACK

@hw4_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])

def update_layout(ques):
    if ques == 'q1':
        return q1_layout
    elif ques == 'q2':
        return q2_layout
    elif ques == 'q3':
        return q3_layout

# INITIALIZE SERVER
hw4_app.run_server(
    port=8035,
    host='0.0.0.0'
)

#%%

