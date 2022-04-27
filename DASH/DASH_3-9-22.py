#%% [markdown]
# DATS-6401 - CLASS 3/9/22
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

# from scipy import stats as stats
# import statistics
# import datetime as dt
# from statsmodels.graphics.gofplots import qqplot
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

print("\nIMPORT SUCCESS")


#%%

jsons = dash.Dash('J-SONS')
jsons.layout = html.Div([
    html.Div(html.H1('TEAM ROSTER: HTML.H1')),
    html.Div(html.H2('SUHAS HTML.H2')),
    html.Div(html.H3('RICARDO HTML.H3')),
    html.Div(html.H4('CHIM HTML.H4')),
    html.Div(html.H5('MAAZ HTML.H5')),
    html.Div(html.H6('NATE HTML.H6')),
])

jsons.run_server(
    port = 8100,
    host = '0.0.0.0'
)


#%%

