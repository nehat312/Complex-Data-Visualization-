#%% [markdown]
# DATS-6401 - LAB #5
# Nate Ehat

#%% [markdown]
## DASH ##
## GOOGLE CLOUD PLATFORM ##

#%% [markdown]
# In this LAB, you will learn how to create an app using Dash in Python.
# You will learn how to load data and display as an interactive web base application using Python.
# The dataset that will be used in this LAB is CONVENIENT_global_confirmed_cases.csv” as used in the previous LABs.
# Some of the package that will be used in this LAB are shown below:

# Create a dashboard with multiple tabs that each tap accommodates each question in this LAB.
# The final python file needs to be deployed through Google cloud (GCP)
# Working link must be provided in the report for grading.

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
# 1. Using Dash in python write an app that plot the COVID global confirmed cases for the following countries:
# (same dataset as LAB  # 3)
    # a. US
    # b. Brazil
    # c. United Kingdom_sun
    # d. China_sum
    # e. India
    # f. Italy
    # g. Germany

# Hint: You need to develop a dropdown menu with the list of countries.
# Make sure to add a title to your app. Use the external style sheet as follow.
# Add the title to the drop dropdown menu as: “Pick the country Name”.

#%%
# IMPORT DATA
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
# 2. Create an app using Dash that plots:
# Quadratic function 𝑓(𝑥) = 𝑎𝑥2 + 𝑏𝑥 + 𝑐 for x between - 2, 2 with 1000 samples.
# The a, b and c must be an input with a slider component.
# Add an appropriate title and label to each section of your app.
# Use the same external style sheet as question 1.




#%%
# 3. Create a calculator app using Dash that perform the following basic math operations:
    # a. Addition
    # b. Subtraction
    # c. Multiplication
    # d. Division
    # e. Log
    # f. Square
    # g. Root square

# Hint: The arithmetic operation must be inside a dropdown menu.
# The input a and input b must be entered through an input field.
# Add an appropriate title and label to each section of your app.
# Use the same external style sheet as question 1.

#%%
# 4. Develop an interactive web-based app using Dash in python to plot a histogram plot for the gaussian distribution.
    # The mean, std, number of samples, number of bins must be entered through a separate slider bar (one slider for each).
    # The mean ranges between -2, 2 with step size of 1.
    # The std ranges from 1 to 3 with step size of 1.
    # Number of samples ranges from 100-10000 with the step size of 500.
    # The number of bins ranges from 20 to 100 with step size of 10.

#%%

# 5. Develop an interactive web-based app using Dash in python to plot polynomial function by entering order of the polynomial through an input field.
    # For example, if the input entered number to input field is 2, then the function 𝑓(𝑥) = 𝑥2 must be plotted.
    # If the entered number to the input field is 3, then the function 𝑓(𝑥) = 𝑥3 must be plotted.
    # The range of x is -2, 2 with 1000 samples in between.

#%%
# 6. Create a Dataframe as follow:
df_bar = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

# Create a dashboard that displays the group bar plot 2x2.
# Below each plot place a slider with min=0 and max = 20 and step size = 1.
# Add the label with H1 for ‘Hello Dash’ and H5 for the ‘Slider number’ label.
# The final web-based app should look like below:

#%%
