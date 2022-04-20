#%% [markdown]
# DATS-6401 - LAB #6
# Nate Ehat

#%% [markdown]
# In this Lab, you will learn how to convert non-gaussian distributed dataset
# into a gaussian distributed dataset.

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
# IMPORT DATA
df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/CONVENIENT_global_confirmed_cases.csv')
df.dropna(axis=0, how='any')
df.drop(index=df.index[0],
        axis=0,
        inplace=True)

print(df.columns)
print(df.info())


#%%
# 1. Generate a random data (x) with 5000 samples and normal distribution (mean = 0, std = 1).
# Then use np.cumsum to convert the generated normal data into a non-normal distributed data (y).
# Graph the normal (x) and non-normal (y) data set versus number of samples and histogram plot of the normal(x) and non-normal (y) dataset
# 2x2 figure using subplot.
# Number of bins = 100. Figure size = 9,7.
# Add grid and appropriate x-label, y-label, and title to the graph.



#%%

# 2. Perform a K-S Normality test on the x and y dataset [dataset generated in the previous question].
# Display the p-value and statistics of the test for the x and y [a separate test is needed for x and y].
# Interpret the K-S test [Normal or Not Normal with 99% accuracy] by looking at the p-value.
# Display the following information on the console:
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  x dataset looks ______
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  y dataset looks ______


#%%

# 3. Repeat Question 2 with the “Shapiro test”.
    # Shapiro test: statistics= _____ p-value = ______
    # Shapiro test: x dataset looks ______
    # Shapiro test: statistics= _____ p-value = ______
    # Shapiro test: y dataset looks ______


#%%

# 4. Repeat Question 2 with the “D'Agostino's 𝐾2 test”.
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______


#%%
# 5. Convert the non-normal data (y) to normal using the rankdata and norm.ppf.
# Add appropriate x- label, y-label, title, and grid to the 2x2 subplot graph.
# The final graph should look like bellow.

#%%
# 6. Plot the QQ plot of the y and the y transformed.
# The final plot should be like below.


#%%
# 7. Perform a K-S Normality test on the y transformed dataset.
# Display the p-value and statistics of the test for y transformed.
# Interpret the K-S test [Normal or Not Normal with 99% accuracy] by looking at p-value.
# Display the following information on the console:
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  y transformed dataset looks ______



#%%
# 8. Repeat Question 7 with the “Shapiro test”.
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______



#%%
# 9. Repeat Question 7 with the “D'Agostino's 𝐾2 test"
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______



#%%
# 10. Do all 3-normality tests confirm the Normality of the transformed data?
# Explain your answer if there is a discrepancy.

#%% [markdown]




#%%

#%%