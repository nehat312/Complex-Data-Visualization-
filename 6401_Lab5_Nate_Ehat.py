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
# DEFINE APP
x = np.linspace(-2, 2, 1000)

lab5_app = dash.Dash('DATA VIZ LAB 5', external_stylesheets=external_stylesheets)
lab5_app.layout = html.Div([html.H1('Lab 5', style={'textAlign': 'center'}),
                          html.Br(),
                          html.Br(),
                          dcc.Tabs(id='hw-questions',
                                   children=[
                                       dcc.Tab(label='QUESTION 1', value='q1'),
                                       dcc.Tab(label='QUESTION 2', value='q2'),
                                       dcc.Tab(label='QUESTION 3', value='q3'),
                                       dcc.Tab(label='QUESTION 4', value='q4'),
                                       dcc.Tab(label='QUESTION 5', value='q5'),
                                       dcc.Tab(label='QUESTION 6', value='q6')]),
                          html.Div(id='layout')])

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

print(df.columns)
print(df.info())

#%%
col_name = []
df['China_sum'] = df.iloc[0:,57:90].astype(float).sum(axis=1)
df['UK_sum'] = df.iloc[0:,249:260].astype(float).sum(axis=1)
#print(df.columns)

for col in df.columns:
    col_name.append(col)
df_covid = df[col_name]
df_covid['date'] = pd.date_range(start='1-23-20', end='11-22-20')

#%%

q1_layout = html.Div([
    dcc.Graph(id='covid-graph'),
    html.Br(),
    html.P('SELECT COUNTRY NAME'),
    dcc.Dropdown(id='country',
                 options=[{'label':'US', 'value':'US'},
                          {'label':'Brazil', 'value':'Brazil'},
                          {'label':'China_sum', 'value':'China_sum'},
                          {'label':'UK_sum', 'value':'UK_sum'},
                          {'label':'Germany', 'value':'Germany'},
                          {'label':'India', 'value':'India'},
                          {'label':'Italy', 'value':'Italy'},
                          ], value='US', clearable=False)

])

@lab5_app.callback(
    Output(component_id='covid-graph', component_property='figure'),
    Input(component_id='country', component_property='value'),
)

def display_chart(country):
    fig = px.line(df_covid, x='date', y=[country])
    return fig

#q1_layout.run_server(
#    port = 8035,
#    host = '0.0.0.0',
#)


# 2. Create an app using Dash that plots:
# Quadratic function 𝑓(𝑥) = 𝑎𝑥2 + 𝑏𝑥 + 𝑐 for x between - 2, 2 with 1000 samples.
# The a, b and c must be an input with a slider component.
# Add an appropriate title and label to each section of your app.
# Use the same external style sheet as question 1.

q2_layout = html.Div([html.H1('QUADRATIC FUNCTION', style={'textAlign': 'Center'}),
                             html.Br(),
                             dcc.Graph(id='quad-graph'),
                             html.P('A:'),
                             dcc.Slider(id='a2',
                                        min=-10,
                                        max=10,
                                        value=1,
                                        marks={f'{i}':i for i in range(-10,11)}),
                             html.Br(),
                             html.P('B:'),
                             dcc.Slider(id='b2',
                                        min=-10,
                                        max=10,
                                        value=2,
                                        marks={f'{i}':i for i in range(-10,11)}),
                             html.Br(),
                             html.P('C:'),
                             dcc.Slider(id='c2',
                                        min=-20,
                                        max=0,
                                        value=3,
                                        marks={f'{i}':i for i in range(-20,1)}),
                             html.Br(),
                             html.Br(),
                             html.Br(),
                             ])


@lab5_app.callback([Output(component_id='quad-graph', component_property='figure'),
                  Output(component_id='func2', component_property='children')],
                 [Input(component_id='a2', component_property='value'),
                  Input(component_id='b2', component_property='value'),
                  Input(component_id='c2', component_property='value')])

def update_q2(a, b, c):
    d = np.sqrt(b**2 - 4*a*c)
    if d < 0:
        raise PreventUpdate
    else:
        y = np.array([a*i**2 + b*i + c for i in x])
        fig = px.line(x=x, y=y)
        disp = f'y(x) = {a}x^2 + {b}x + {c}'
        return fig, disp


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

q3_layout = html.Div([
    html.H1('Calculator', style={'textAlign': 'Center'}),
    html.Br(),
    html.P('Please enter the first number:'),
    html.Div(['Input:',
              dcc.Input(id='inputA3', value=0, type='float')]),
    html.Br(),
    dcc.Dropdown(id='sign3',
                 options=[{'label': 'Addition', 'value': '+'},
                          {'label': 'Subtraction', 'value': '-'},
                          {'label': 'Multiplication', 'value': '*'},
                          {'label': 'Division', 'value': '/'},
                          {'label': 'Log', 'value': 'log'},
                          {'label': 'Power', 'value': '^'},
                          {'label': 'Root', 'value': 'root'}],
                 value='+'),
    html.Br(),
    html.P('Please enter the second number:'),
    html.Div(['Input:',
              dcc.Input(id='inputB3', value=0, type='float')]),
    html.Br(),
    html.Br(),
    html.Br(),
    html.H1(id='output3')
])

@lab5_app.callback(Output(component_id='output3', component_property='children'),
                 [Input(component_id='inputA3', component_property='value'),
                  Input(component_id='sign3', component_property='value'),
                  Input(component_id='inputB3', component_property='value'),
                  ])

def update_q3(a, op, b):
    if op == '+':
        r = float(a) + float(b)
    elif op == '-':
        r = float(a) - float(b)
    elif op == '*':
        r = float(a) * float(b)
    elif op == '/':
        if float(b) == 0:
            return 'Cannot divide by zero.'
        else:
            r = float(a) / float(b)
    elif op == 'log':
        r = float(a) * np.log10(float(b))
    elif op == '^':
        r = float(a) ** float(b)
    elif op == 'root':
        r = np.sqrt((float(a) ** 2) + (float(b) ** 2))
        return f'Root of {a} square + {b} square = {r}'
    return f'{a} {op} {b} = {r}'


# 4. Develop an interactive web-based app using Dash in python to plot a histogram plot for the gaussian distribution.
    # The mean, std, number of samples, number of bins must be entered through a separate slider bar (one slider for each).
    # The mean ranges between -2, 2 with step size of 1.
    # The std ranges from 1 to 3 with step size of 1.
    # Number of samples ranges from 100-10000 with the step size of 500.
    # The number of bins ranges from 20 to 100 with step size of 10.

q4_layout = html.Div([
    html.H1('Gaussian Distribution', style={'textAlign': 'Center'}),
    html.Br(),
    dcc.Graph(id='graph4'),
    html.P('Mean:'),
    dcc.Slider(id='mean4',
               min=-2,
               max=2,
               value=0,
               marks={'-2': 2, '-1': -1, '0': 0, '1': 1, '2': 2}),
    html.Br(),
    html.P('Standard Deviation:'),
    dcc.Slider(id='std4',
               min=1,
               max=3,
               value=1,
               marks={'1': 1, '2': 2, '3': 3}),
    html.Br(),
    html.P('Sample Size:'),
    dcc.Slider(id='size4',
               min=100,
               max=10000,
               value=100,
               step=500,
               marks={f'{i}':i for i in range(100,10001,500)}),
    html.Br(),
    html.P('Number of Bins:'),
    dcc.Slider(id='bins4',
               min=20,
               max=100,
               value=50,
               step=10,
               marks={f'{i}':i for i in range(20,101,10)}),
    html.Br(),
    html.Br(),
    html.Br(),
])

@lab5_app.callback(Output(component_id='graph4', component_property='figure'),
                 [Input(component_id='mean4', component_property='value'),
                  Input(component_id='std4', component_property='value'),
                  Input(component_id='size4', component_property='value'),
                  Input(component_id='bins4', component_property='value'),
                  ])

def update_q4(m, s, n, b):
    x4 = np.random.normal(m, s, n)
    fig = px.histogram(x=x4, nbins=b, range_x=[-5, 5])
    return fig

# 5. Develop an interactive web-based app using Dash in python to plot polynomial function by entering order of the polynomial through an input field.
    # For example, if the input entered number to input field is 2, then the function 𝑓(𝑥) = 𝑥2 must be plotted.
    # If the entered number to the input field is 3, then the function 𝑓(𝑥) = 𝑥3 must be plotted.
    # The range of x is -2, 2 with 1000 samples in between.

q5_layout = html.Div([html.H1('Polynomial Function', style={'textAlign': 'Center'}),
                             html.Br(),
                             html.P('Enter degree of polynomial:'),
                             html.Br(),
                             dcc.Input(id='input5'),
                             html.Br(),
                             html.Br(),
                             html.H3(id='func5'),
                             html.Br(),
                             html.Br(),
                             dcc.Graph(id='graph5')
                             ])

@lab5_app.callback([Output(component_id='graph5', component_property='figure'),
                  Output(component_id='func5', component_property='children')],
                 Input(component_id='input5', component_property='value'))

def update_q5(n):
    y = np.array([np.power(i,int(n)) for i in x])
    fig = px.line(x=x, y=y)
    disp = f'y(x) = x^{n}'
    return fig, disp

# 6. Create a Dataframe as follows:
    # Create a dashboard that displays the group bar plot 2x2.
    # Below each plot place a slider with min=0 and max = 20 and step size = 1.
    # Add the label with H1 for ‘Hello Dash’ and H5 for the ‘Slider number’ label.
    # The final web-based app should look like below:

df_fruits = pd.DataFrame({'Fruit':['Apples','Oranges','Bananas','Apples','Oranges','Bananas'],
                    'Amount':[4,1,2,2,4,5],
                    'City':['SF','SF','SF','Montreal','Montreal','Montreal']})

fig = px.bar(df_fruits, x='Fruit', y='Amount', color='City', barmode='group')
subplot_1x1 = html.Div([html.H1('Hello Dash 1'),
                        html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='graph6_1',figure=fig),
                        html.H3('Slider 1'),
                        dcc.Slider(id='s1',
                                   min=0,
                                   max=20,
                                   value=10)
                        ])

subplot_1x2 = html.Div([html.H1('Hello Dash 2'),
                        html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='graph6_2',figure=fig),
                        html.H3('Slider 2'),
                        dcc.Slider(id='s2',
                                   min=0,
                                   max=20,
                                   value=10)
                        ])
subplot_2x1 = html.Div([html.H1('Hello Dash 3'),
                        html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='graph6_3',figure=fig),
                        html.H3('Slider 3'),
                        dcc.Slider(id='s3',
                                   min=0,
                                   max=20,
                                   value=10)
                        ])
subplot_2x2 = html.Div([html.H1('Hello Dash 4'),
                        html.P('Dash: A web application framework for Python.'),
                        dcc.Graph(id='graph6_4',figure=fig),
                        html.H3('Slider 4'),
                        dcc.Slider(id='s4',
                                   min=0,
                                   max=20,
                                   value=10)
                        ])
q6_layout = html.Div([html.Div(subplot_1x1,style={'width':'49%','display':'inline-block'}),
                             html.Div(subplot_1x2,style={'width':'49%','display':'inline-block'}),
                             html.Div(subplot_2x1,style={'width':'49%','display':'inline-block'}),
                             html.Div(subplot_2x2,style={'width':'49%','display':'inline-block'})])


@lab5_app.callback(Output(component_id='layout', component_property='children'),
                 [Input(component_id='hw-questions', component_property='value')])

def update_layout(ques):
    if ques == 'q1':
        return q1_layout
    elif ques == 'q2':
        return q2_layout
    elif ques == 'q3':
        return q3_layout
    elif ques == 'q4':
        return q4_layout
    elif ques == 'q5':
        return q5_layout
    elif ques == 'q6':
        return q6_layout

# INITIALIZE SERVER
lab5_app.run_server(
    port=8035,
    host='0.0.0.0'
)

#%%
