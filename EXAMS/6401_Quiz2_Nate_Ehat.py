#%% [markdown]
# DATS-6401 - QUIZ #2
# Nate Ehat

#%% [markdown]
# 1. Using the Dash in python develop an App that plot a pie chart for the ‘tips’ dataset interactively.
# The App should satisfy the following criterions:
    # a. Create a dropdown menu that a user can select the feature inside the ‘tips’ dataset.
    # Select the label for this dropdown menu as : “Please select the feature from the menu”.
    # The features are:
        # i. Day
        # ii. Time
        # iii. Sex
    # b. Create another drop-down menu that a user can select the output for the plot.
    # Select the label for this drop-down menu as : “Please select the output variable to be plotted”.
    # The output variables are:
        # i. total_bill
        # ii. tip
        # iii. size
    # c. Based on the information in a and b then plot a pie plot accordingly.
    # The final app should be like the following. For the submission only the .py is required.

# Extra credit : GCP deployment +10 points. You need to include the working google cloud link into your submission.


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
# IMPORT DATA
tips = px.data.tips()
print(tips.info())
print(tips.columns)

#%%
# Assign app name
tips_app = dash.Dash('TIPS APP', external_stylesheets=external_stylesheets)

#server = covid_app.server

# Define app layout
tips_app.layout = html.Div([
    dcc.Graph(id='tips-graph'),
    html.P('PLEASE SELECT INPUT FEATURE:'),
    dcc.Dropdown(id='dropdown1',
                 options=[{'label':'day', 'value':'day'},
                          {'label':'time', 'value':'time'},
                          {'label':'sex', 'value':'sex'},
                          ], value='day', clearable=False),
    html.Br(),
    html.P('PLEASE SELECT OUTPUT VARIABLE:'),
    dcc.Dropdown(id='dropdown2',
                 options=[{'label':'total_bill', 'value':'total_bill'},
                          {'label':'tip', 'value':'tip'},
                          {'label':'size', 'value':'size'},
                          ], value='total_bill', clearable=False),
])

@tips_app.callback(
    Output(component_id='tips-graph', component_property='figure'),
    Input(component_id='dropdown1', component_property='value'),
    Input(component_id='dropdown2', component_property='value'),
)

def display_chart(dropdown1, dropdown2):
    fig = px.pie(tips, values=dropdown2, names=dropdown1, title='TIPS PIE CHART') # names=[input], #labels=tips[dropdown1],values=[dropdown2]
    return fig

tips_app.run_server(
    port = 8030,
    host = '0.0.0.0',
)

#%%