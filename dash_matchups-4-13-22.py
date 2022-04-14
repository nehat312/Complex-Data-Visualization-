#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import dash as dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly as ply
import plotly.express as px

print("\nIMPORT SUCCESS")

#%%
tr_filepath = 'drive/My Drive/GWU/TEAM-7/data/tr_data_hub_4-13-22'

tr_data_hub = pd.read_excel(tr_filepath + '.xlsx', index_col='Team')
#tr_data_hub = pd.read_csv(tr_filepath + '.csv', index_col='Team')
#tr_data_hub.info()


cbs_filepath = 'drive/My Drive/GWU/TEAM-7/data/cbs-matchups-4-13-22'

cbs_df = pd.read_excel(cbs_filepath + '.xlsx', index_col='Unnamed: 0')
#cbs_df = pd.read_csv(cbs_filepath + '.csv', index_col='Unnamed: 0')
#cbs_df.drop(columns=['Unnamed: 0'], inplace=True)
print(cbs_df)

#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
tr_app = dash.Dash('TEAM RANKINGS APP', external_stylesheets=external_stylesheets)
tr_chart = html.Div([html.H1('NBA DATA - 2022 YTD', style={'textAlign': 'Center'}),
                     html.Br(),
                     html.P('TBU'),
                     dcc.Graph(id='tr-chart'),
                     html.Br(),
                     html.P('STAT A'),
                     dcc.Dropdown(id='stat-a',
                                  options=[{'label': 'effective-field-goal-pct', 'value': 'effective-field-goal-pct'},
                                           {'label': 'true-shooting-percentage', 'value': 'true-shooting-percentage'},
                                           {'label': 'offensive-efficiency', 'value': 'offensive-efficiency'},
                                           {'label': 'defensive-efficiency', 'value': 'defensive-efficiency'},
                                           {'label': 'net-adj-efficiency', 'value': 'net-adj-efficiency'},
                                           {'label': 'three-pointers-made-per-game', 'value': 'three-pointers-made-per-game'},
                                           {'label': 'three-pointers-attempted-per-game', 'value': 'three-pointers-attempted-per-game'},
                                           {'label': 'assist--per--turnover-ratio', 'value': 'assist--per--turnover-ratio'}], value='defensive-efficiency'),
                     html.Br(),
                     html.P('STAT B'),
                     dcc.Dropdown(id='stat-b',
                                  options=[{'label': 'effective-field-goal-pct', 'value': 'effective-field-goal-pct'},
                                           {'label': 'true-shooting-percentage', 'value': 'true-shooting-percentage'},
                                           {'label': 'offensive-efficiency', 'value': 'offensive-efficiency'},
                                           {'label': 'defensive-efficiency', 'value': 'defensive-efficiency'},
                                           {'label': 'net-adj-efficiency', 'value': 'net-adj-efficiency'},
                                           {'label': 'three-pointers-made-per-game', 'value': 'three-pointers-made-per-game'},
                                           {'label': 'three-pointers-attempted-per-game', 'value': 'three-pointers-attempted-per-game'},
                                           {'label': 'assist--per--turnover-ratio', 'value': 'assist--per--turnover-ratio'}], value='defensive-efficiency'),
                     html.Br(),
                     ])

@tr_app.callback(Output(component_id='tr-chart', component_property='figure'),
                  #Output(component_id='func2', component_property='children')],
                 [Input(component_id='stat-a', component_property='value'),
                  Input(component_id='stat-b', component_property='value')])

def display_chart(stata, statb):
    fig = px.scatter(tr_data_hub, x=[stata], y=[statb])
    return fig

tr_app.run_server(
    port = 8030,
    host = '0.0.0.0',
)



#%%



#%%