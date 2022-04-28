#%%
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import dash as dash
from dash import dash_table
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate

import plotly as ply
import plotly.express as px

import requests
from bs4 import BeautifulSoup
import re

import datetime
import sys
import os

from scipy import stats as stats
import statistics

import json
import time
import nltk

print("\nIMPORT SUCCESS")

#%%
# CLEAN DATA IMPORT
tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
kp_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/kenpom_pull_3-14-22'

# TEAMRANKINGS DATA
tr_df = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'
#tr_df = pd.read_csv(mtr_filepath + '.csv')

# KENPOM DATA
kp_df = pd.read_excel(kp_filepath + '.xlsx') #index_col='Team'
#kp_df = pd.read_csv(kp_filepath + '.csv')

tr_df = tr_df.round(2)
#tr_df = pd.to_numeric(tr_df)

print(tr_df.head())

#%%
print(tr_df.info())
#print(kp_df.info())
#print(tr_df.index)
#print(tr_df)

#%%
tr_df = tr_df[['Team', 'win-pct-all-games',
               'average-scoring-margin', 'opponent-average-scoring-margin',
               'points-per-game', 'opponent-points-per-game',
               'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
               'effective-field-goal-pct', 'true-shooting-percentage',
               'three-point-pct', 'two-point-pct', 'free-throw-pct',
               'opponent-effective-field-goal-pct', 'opponent-true-shooting-percentage',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'assists-per-game', 'turnovers-per-game',
               'opponent-assists-per-game', 'opponent-turnovers-per-game',
               'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
               'blocks-per-game', 'steals-per-game', 'stocks-per-game',
               'opponent-blocks-per-game','opponent-steals-per-game'
               ]]


print(tr_df.info())


#%%
# RENAME COLUMNS TO IMPROVE APP OPTICS
app_cols = {'Team': 'TEAM', 'win-pct-all-games':'WIN%',
            'average-scoring-margin':'AVG_MARGIN', 'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'points-per-game': 'PTS/GM',  'opponent-points-per-game':'OPP_PTS/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF', 'net-adj-efficiency':'NET_ADJ_EFF',
            'effective-field-goal-pct':'EFG%', 'true-shooting-percentage':'TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'opponent-effective-field-goal-pct':'OPP_EFG%', 'opponent-true-shooting-percentage':'OPP_TS%',
            'assists-per-game':'AST/GM', 'turnovers-per-game':'TO/GM', 'assist--per--turnover-ratio':'AST/TO',
            'opponent-assists-per-game':'OPP_AST/GM', 'opponent-turnovers-per-game':'OPP_TO/GM', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'blocks-per-game':'B/GM', 'steals-per-game':'S/GM', 'stocks-per-game':'STOCKS/GM',
            'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM',# 'opponent-stocks-per-game':'OPP_STL/GM',
            }


tr_cols = {'Team': 'TEAM', 'points-per-game':'PTS/GM', 'average-scoring-margin':'AVG_MARGIN', 'win-pct-all-games':'WIN%', 'win-pct-close-games':'WIN%_CLOSE',
            'effective-field-goal-pct':'EFG%', 'true-shooting-percentage':'TS%', 'effective-possession-ratio': 'POSS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'field-goals-made-per-game':'FGM/GM', 'field-goals-attempted-per-game':'FGA/GM', 'three-pointers-made-per-game':'3PM/GM', 'three-pointers-attempted-per-game':'3PA/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF',
            'total-rebounds-per-game':'TRB/GM', 'offensive-rebounds-per-game':'ORB/GM', 'defensive-rebounds-per-game':'DRB/GM',
            'offensive-rebounding-pct':'ORB%', 'defensive-rebounding-pct':'DRB%', 'total-rebounding-percentage':'TRB%',
            'blocks-per-game':'B/GM', 'steals-per-game':'S/GM', 'assists-per-game':'AST/GM', 'turnovers-per-game':'TO/GM',
            'assist--per--turnover-ratio':'AST/TO', 'possessions-per-game':'POSS/GM', 'personal-fouls-per-game':'PF/GM',
            'opponent-points-per-game':'OPP_PTS/GM', 'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'opponent-effective-field-goal-pct':'OPP_EFG%', 'opponent-true-shooting-percentage':'OPP_TS%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%', 'opponent-shooting-pct':'OPP_FG%',
            'opponent-assists-per-game':'OPP_AST/GM', 'opponent-turnovers-per-game':'OPP_TO/GM', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'opponent-offensive-rebounds-per-game':'OPP_OREB/GM', 'opponent-defensive-rebounds-per-game':'OPP_DREB/GM', 'opponent-total-rebounds-per-game':'OPP_TREB/GM',
            'opponent-offensive-rebounding-pct':'OPP_OREB%', 'opponent-defensive-rebounding-pct':'OPP_DREB%',
            'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM',
            'opponent-effective-possession-ratio':'OPP_POSS%',
            'net-avg-scoring-margin':'NET_AVG_MARGIN', 'net-points-per-game':'NET_PTS/GM',
            'net-adj-efficiency':'NET_ADJ_EFF',
            'net-effective-field-goal-pct':'NET_EFG%', 'net-true-shooting-percentage':'NET_TS%',
            'stocks-per-game':'STOCKS/GM', 'total-turnovers-per-game':'TTL_TO/GM',
            'net-assist--per--turnover-ratio':'NET_AST/TO',
            'net-total-rebounds-per-game':'NET_TREB/GM', 'net-off-rebound-pct':'NET_OREB%', 'net-def-rebound-pct':'NET_DREB%'
            }


#%%
#tr_df['VISITOR_CODE'] = matchup_history['VISITOR'].map(team_code_dict)
tr_df.columns = tr_df.columns.map(app_cols)

print(tr_df.columns)
print(tr_df.info())

#%%

def discrete_background_color_bins(tr_df, n_bins=5, columns='all'):
    import colorlover
    bounds = [i * (1.0 / n_bins) for i in range(n_bins + 1)]
    if columns == 'all':
        if 'id' in tr_df:
            df_numeric_columns = tr_df.select_dtypes('number').drop(['id'], axis=1)
        else:
            df_numeric_columns = tr_df.select_dtypes('number')
    else:
        df_numeric_columns = tr_df[columns]
    df_max = df_numeric_columns.max().max()
    df_min = df_numeric_columns.min().min()
    ranges = [
        ((df_max - df_min) * i) + df_min
        for i in bounds
    ]
    styles = []
    legend = []
    for i in range(1, len(bounds)):
        min_bound = ranges[i - 1]
        max_bound = ranges[i]
        backgroundColor = colorlover.scales[str(n_bins)]['seq']['Blues'][i - 1]
        color = 'white' if i > len(bounds) / 2. else 'inherit'

        for column in df_numeric_columns:
            styles.append({
                'if': {
                    'filter_query': (
                        '{{{column}}} >= {min_bound}' +
                        (' && {{{column}}} < {max_bound}' if (i < len(bounds) - 1) else '')
                    ).format(column=column, min_bound=min_bound, max_bound=max_bound),
                    'column_id': column
                },
                'backgroundColor': backgroundColor,
                'color': color
            })
        legend.append(
            html.Div(style={'display': 'inline-block', 'width': '60px'}, children=[
                html.Div(
                    style={
                        'backgroundColor': backgroundColor,
                        'borderLeft': '1px rgb(50, 50, 50) solid',
                        'height': '10px'
                    }
                ),
                html.Small(round(min_bound, 2), style={'paddingLeft': '2px'})
            ])
        )

    return (styles, html.Div(legend, style={'padding': '5px 0 5px 0'}))

(styles, legend) = discrete_background_color_bins(tr_df)

#print(styles)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
ncaab_app = dash.Dash('NCAAM BASKETBALL DASHBOARD') #, external_stylesheets=external_stylesheets
application = ncaab_app.server

#app = Dash(__name__)

# {current_time:%Y-%m-%d %H:%M}
#'backgroundColor': 'rgb(220, 220, 220)',

ncaab_app.layout = html.Div([html.H1('NCAAM BASKETBALL DASHBOARD', style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '4px solid black'}),
                     #html.Br(),
                     # html.Br(),
                          dcc.Tabs(id='ncaa-tabs',
                                   children=[
                                       dcc.Tab(label='TEAM-VIZ', value='team-viz'),
                                       dcc.Tab(label='STAT-VIZ', value='stat-viz'),
                                       dcc.Tab(label='CAT-VIZ', value='cat-viz'),

                               dash_table.DataTable(tr_df.to_dict('records'),
                                                    columns=[{"name": i, "id": i} for i in tr_df.columns],
                                                    id='tr-df',
                                                    style_data={'textAlign': 'Center', 'fontWeight': 'bold', 'border': '2px solid black'},
                                                    style_cell={'textAlign': 'Center', 'fontWeight': 'bold', 'padding': '5px'},   #324f6e - TBD  #B10DC9 - fuschia #7FDBFF - Aqua
                                                    style_header={'backgroundColor': '#7FDBFF', 'color': 'black', 'fontWeight': 'bold', 'border': '2px solid black'}, #'1px solid blue'
                                                    sort_action='native',
                                                    style_data_conditional = [styles]

                                                    ),
                                         ]),
                               html.Br(),
                               html.P('METRIC COMPARISON'),
                               dcc.Graph(id='chart'),
                               html.Br(),
                               html.P('STAT A'),
                               dcc.Dropdown(id='stata',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     #{'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     #{'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     #{'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     #{'label': 'TREB%', 'value': 'TREB%'},
                                                     #{'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     #{'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},
                                                    ], value='WIN%'
                                            ),
                               html.Br(),
                               html.P('STAT B'),
                               dcc.Dropdown(id='statb',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     #{'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     #{'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     #{'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     #{'label': 'TREB%', 'value': 'TREB%'},
                                                     #{'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     #{'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},
                                                    ], value='WIN%'
                                            ),
                               html.Br(),

                               html.Div(id='dashboard-layout')])

@ncaab_app.callback(Output(component_id='layout', component_property='children'),
                      #Output(component_id='tr-df', component_property='figure'),
                      #Output(component_id='chart', component_property='figure'),
                      #[Input(component_id='stata', component_property='value'),
                       #Input(component_id='statb', component_property='value')]
 )

def display_dataframe(dataframe):
    return dataframe

@ncaab_app.callback(Output(component_id='chart', component_property='figure'),
                    [Input(component_id='stata', component_property='value'),
                     Input(component_id='statb', component_property='value')])


def display_chart(statb): #stata,
    fig = px.scatter(tr_df, x='WIN%', y=[statb])
    return fig

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)


#%%

#{'if': {'row_index': [2,3,6,7,10,11]}, 'backgroundColor': 'rgb(211, 211, 211)'},
#{'if': {'filter_query': '{FINAL} > 105', 'column_id': 'FINAL', 'row_index': [0,1]},
#'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red':'#2ECC40', gray?: '#FF4136'

##{'if': {'filter_query': '{MODEL_PREDICTION} = "LOSS"','column_id': 'MODEL_PREDICTION'},
#'color': 'black', 'backgroundColor': '#FF4136', 'fontWeight': 'bold'}, # RED '#FF4136' / GREEN '#2ECC40'
# {'if': {'column_id': 'MODEL_PREDICTION'}, 'fontWeight': 'bold',},], #'#01FF70'  'backgroundColor': '#01FF70',

#%%

# SCATTERPLOT -
plt.figure(figsize=(10,10))
sns.scatterplot(data=tr_df, x='offensive-efficiency', y='defensive-efficiency', palette='magma', markers=True)
plt.title('OFF. VS. DEF. EFFICIENCY (BY CONFERENCE)', fontsize=16)
plt.xlabel('OFFENSIVE EFFICIENCY', fontsize=16)
plt.ylabel('DEFENSIVE EFFICIENCY', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

for i in tr_df.index:
  plt.text(tr_df['offensive-efficiency'][tr_data_hub.index==i]+.01,tr_data_hub['defensive-efficiency'][tr_data_hub.index==i]+.01,str(i), color='black')

plt.show()




#%%

## DATA IMPORT

# kenpom_2022 = pd.read_csv('drive/My Drive/SPORTS/kenpom_pull_3-14-22.csv')
kenpom_2022 = pd.read_excel('drive/My Drive/SPORTS/kenpom_pull_3-14-22.xlsx')

kenpom_2022 = kenpom_2022[kenpom_2022['Year'] == 2022]
kenpom_2022 = kenpom_2022.drop(
    columns=['Year', 'AdjO Rank', 'AdjD Rank', 'AdjT Rank', 'SOS OppO Rank', 'SOS OppD Rank', 'SOS Adj EM Rank',
             'NCSOS Adj EM Rank', 'Luck Rank'])
mm_2022 = kenpom_2022[kenpom_2022['Seed'] >= 1]
mm_2022 = mm_2022.set_index('Team')
# print(kenpom_2022.info())
# print(kenpom_2022.head())

print(mm_2022.info())
print(mm_2022.head())

#%%

kp_cols = kenpom_2022.columns
mm_cols = mm_2022.columns
kp_num_cols = kenpom_2022[
    ['Rank', 'Win', 'Loss', 'Seed', 'Adj EM', 'AdjO', 'AdjD', 'AdjT', 'SOS Adj EM', 'SOS OppO', 'SOS OppD',
     'NCSOS Adj EM']]
kp_cat_cols = kenpom_2022[['Team', 'Conference']]

# top50_2022 = kenpom_2022[(kenpom_2022['Rank'] <= 50)].drop(columns=['Year'])
# top100_2022 = kenpom_2022[(kenpom_2022['Rank'] <= 100)].drop(columns=['Year'])
# top150_2022 = kenpom_2022[(kenpom_2022['Rank'] <= 150)].drop(columns=['Year'])

# print(kp_num_cols)
# kenpom_2022[:100]
print(mm_cols)
print('---------------------------------------------------------------------------------')
print(mm_2022)

print(mm_2022.index)

#%%
# CORRELATION MATRIX
mm_2022.corr()


#%%
## DATA VIZ

# create correlation variables relative to rest of DataFrame
rank_corr = mm_2022.corr()[['Rank']].sort_values(by='Rank', ascending=False)
seed_corr = mm_2022.corr()[['Seed']].sort_values(by='Seed', ascending=False)

# create heatmap to visualize correlation variable
# SUBPLOTS
plt.figure(figsize=(10, 8))
sns.heatmap(rank_corr, annot=True, cmap='flare', vmin=-1, vmax=1, linecolor='white', linewidth=2);
# sns.heatmap(seed_corr, annot = True, cmap = 'flare', vmin=-1, vmax=1, linecolor = 'white', linewidth = 2);

# SCATTERPLOT - NET EM VS WINS

# f, (ax1) = plt.subplots(1, 1)

# plt.figure(figsize=(16,8))

plt.subplot(1, 2, 1)
sns.scatterplot(data=mm_2022, x='AdjO', y='AdjD', hue='Conference', style='Conference', size='Conference',
                palette='magma', markers=True)
plt.title('AdjO vs. AdjD (BY CONFERENCE)', fontsize=16)
# sns.set_title('AdjO vs. AdjD ')
# plt.xlabel('ADJ. OFFENSE', fontsize=16)
# plt.ylabel('ADJ. DEFENSE', fontsize=16)
plt.axis('square')
plt.legend(loc='best')

for i in mm_2022.index:
    plt.text(mm_2022.AdjO[mm_2022.index == i] + .01, mm_2022.AdjD[mm_2022.index == i] + .01, str(i), color='black')

plt.subplot(1, 2, 2)
sns.scatterplot(data=mm_2022, x='Adj EM', y='Seed', hue='Conference', style='Conference', size='Conference',
                palette='magma', markers=True)
plt.title('ADJ EM vs. SEED', fontsize=16)
# sns.set_title('AdjO vs. AdjD ')
# plt.xlabel('ADJ. OFFENSE', fontsize=16)
# plt.ylabel('ADJ. DEFENSE', fontsize=16)

plt.axis('square')
# ax.axis('equal')
plt.legend(loc='best')

for i in mm_2022.index:
    plt.text(mm_2022.AdjO[mm_2022.index == i] + .01, mm_2022.AdjD[mm_2022.index == i] + .01, str(i), color='black')

# plt.grid()
plt.tight_layout(pad=1)

plt.show()


#%%

tourney_teams = tr_df.loc[['Gonzaga', 'Arizona', 'Kansas', 'Baylor', #1
                           'Duke', 'Kentucky', 'Auburn', 'Villanova', #2
                                        'Purdue', 'Texas Tech',  'Tennessee', 'Wisconsin', #3
                                        'Arkansas', 'UCLA', 'Illinois', 'Providence', #4
                                        'Iowa', 'Houston', 'Connecticut', 'St Marys', #5
                                        'Alabama', 'Colorado St', 'LSU', 'Texas', #6
                                        'Murray St', 'USC', 'Michigan St', 'Ohio State', #7
                                        'N Carolina', 'Boise State', 'San Diego St', 'Seton Hall', #8
                                        'Memphis', 'Creighton', 'Marquette', 'TX Christian',  #9
                                        'San Francisco', 'Miami (FL)', 'Davidson', 'Loyola-Chi', #10
                                        'Notre Dame', 'Michigan', 'VA Tech',  'Rutgers', 'Iowa State', #11
                                        'Indiana', 'Wyoming', 'Richmond', 'N Mex State', 'UAB', #12
                                        'Vermont', 'S Dakota St', 'Akron', 'Chattanooga',#13
                                        'Montana St', 'Longwood', 'Colgate', 'Yale', #14
                                        'St Peters', 'Jackson St', 'CS Fullerton', 'Delaware', #15
                                        'Georgia St', 'Norfolk St', 'TX Southern', 'Bryant',]] #16

tourney_teams_dict = {1:['Gonzaga',  'Arizona', 'Kansas', 'Baylor'], #1
                      2:['Duke', 'Kentucky', 'Auburn', 'Villanova'], #2
                      3:['Purdue', 'Texas Tech',  'Tennessee', 'Wisconsin'], #3
                     4: ['Arkansas', 'UCLA', 'Illinois', 'Providence'], #4
                                        5: ['Iowa', 'Houston', 'Connecticut', 'St Marys'], #5
                                        6: ['Alabama', 'Colorado St', 'LSU', 'Texas'], #6
                                        7: ['Murray St', 'USC', 'Michigan St', 'Ohio State'], #7
                                        8: ['N Carolina', 'Boise State', 'San Diego St', 'Seton Hall'], #8
                                        9: ['Memphis', 'Creighton', 'Marquette', 'TX Christian'],  #9
                                        10: ['San Francisco', 'Miami (FL)', 'Davidson', 'Loyola-Chi'], #10
                                        11: ['Notre Dame', 'Michigan', 'VA Tech',  'Rutgers', 'Iowa State'], #11
                                        12: ['Indiana', 'Wyoming', 'Richmond', 'N Mex State', 'UAB'], #12
                                        13: ['Vermont', 'S Dakota St', 'Akron', 'Chattanooga'],#13
                                        14: ['Montana St', 'Longwood', 'Colgate', 'Yale'], #14
                                        15: ['St Peters', 'Jackson St', 'CS Fullerton', 'Delaware'], #15
                                        16: ['Georgia St', 'Norfolk St', 'TX Southern', 'Bryant']
                      }

tourney_teams.reset_index(inplace=True)
print(tourney_teams.info())
  #print(tourney_teams)
  #print(tr_data_hub_2022.info())
  #print(tr_data_hub_2022.head())

#%%

print(tourney_teams.columns)
print(tourney_teams_dict.keys())
print(tourney_teams_dict.values())
print(tourney_teams_dict)

#%%

teamlist = ['Gonzaga', 'Arizona', 'Kansas', 'Baylor', #1
                           'Duke', 'Kentucky', 'Auburn', 'Villanova', #2
                                        'Purdue', 'Texas Tech',  'Tennessee', 'Wisconsin', #3
                                        'Arkansas', 'UCLA', 'Illinois', 'Providence', #4
                                        'Iowa', 'Houston', 'Connecticut', 'St Marys', #5
                                        'Alabama', 'Colorado St', 'LSU', 'Texas', #6
                                        'Murray St', 'USC', 'Michigan St', 'Ohio State', #7
                                        'N Carolina', 'Boise State', 'San Diego St', 'Seton Hall', #8
                                        'Memphis', 'Creighton', 'Marquette', 'TX Christian',  #9
                                        'San Francisco', 'Miami (FL)', 'Davidson', 'Loyola-Chi', #10
                                        'Notre Dame', 'Michigan', 'VA Tech',  'Rutgers', 'Iowa State', #11
                                        'Indiana', 'Wyoming', 'Richmond', 'N Mex State', 'UAB', #12
                                        'Vermont', 'S Dakota St', 'Akron', 'Chattanooga',#13
                                        'Montana St', 'Longwood', 'Colgate', 'Yale', #14
                                        'St Peters', 'Jackson St', 'CS Fullerton', 'Delaware', #15
                                        'Georgia St', 'Norfolk St', 'TX Southern', 'Bryant',] #16

#tr_df = [tr_df['Team'][teamlist]]




#%%



tourney_teams = tr_df.loc[['Gonzaga', 'Arizona', 'Kansas', 'Baylor', #1
                           'Duke', 'Kentucky', 'Auburn', 'Villanova', #2
                                        'Purdue', 'Texas Tech',  'Tennessee', 'Wisconsin', #3
                                        'Arkansas', 'UCLA', 'Illinois', 'Providence', #4
                                        'Iowa', 'Houston', 'Connecticut', 'St Marys', #5
                                        'Alabama', 'Colorado St', 'LSU', 'Texas', #6
                                        'Murray St', 'USC', 'Michigan St', 'Ohio State', #7
                                        'N Carolina', 'Boise State', 'San Diego St', 'Seton Hall', #8
                                        'Memphis', 'Creighton', 'Marquette', 'TX Christian',  #9
                                        'San Francisco', 'Miami (FL)', 'Davidson', 'Loyola-Chi', #10
                                        'Notre Dame', 'Michigan', 'VA Tech',  'Rutgers', 'Iowa State', #11
                                        'Indiana', 'Wyoming', 'Richmond', 'N Mex State', 'UAB', #12
                                        'Vermont', 'S Dakota St', 'Akron', 'Chattanooga',#13
                                        'Montana St', 'Longwood', 'Colgate', 'Yale', #14
                                        'St Peters', 'Jackson St', 'CS Fullerton', 'Delaware', #15
                                        'Georgia St', 'Norfolk St', 'TX Southern', 'Bryant',]] #16

#tourney_teams.reset_index(inplace=True)

print(tr_df.columns)
print(tr_df.info())

#%%
tourney_teams_dict = {1:['Gonzaga',  'Arizona', 'Kansas', 'Baylor'], #1
                      2:['Duke', 'Kentucky', 'Auburn', 'Villanova'], #2
                      3:['Purdue', 'Texas Tech',  'Tennessee', 'Wisconsin'], #3
                     4: ['Arkansas', 'UCLA', 'Illinois', 'Providence'], #4
                                        5: ['Iowa', 'Houston', 'Connecticut', 'St Marys'], #5
                                        6: ['Alabama', 'Colorado St', 'LSU', 'Texas'], #6
                                        7: ['Murray St', 'USC', 'Michigan St', 'Ohio State'], #7
                                        8: ['N Carolina', 'Boise State', 'San Diego St', 'Seton Hall'], #8
                                        9: ['Memphis', 'Creighton', 'Marquette', 'TX Christian'],  #9
                                        10: ['San Francisco', 'Miami (FL)', 'Davidson', 'Loyola-Chi'], #10
                                        11: ['Notre Dame', 'Michigan', 'VA Tech',  'Rutgers', 'Iowa State'], #11
                                        12: ['Indiana', 'Wyoming', 'Richmond', 'N Mex State', 'UAB'], #12
                                        13: ['Vermont', 'S Dakota St', 'Akron', 'Chattanooga'],#13
                                        14: ['Montana St', 'Longwood', 'Colgate', 'Yale'], #14
                                        15: ['St Peters', 'Jackson St', 'CS Fullerton', 'Delaware'], #15
                                        16: ['Georgia St', 'Norfolk St', 'TX Southern', 'Bryant']}

tr_df['SEED'] = tr_df['Team'].map(tourney_teams_dict)
print(tr_df['SEED'].isnull().sum())


#%%
import dash
import dash_html_components as html
import base64

app = dash.Dash()


image_filename = '/NBA-logo.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div([
    html.Img(src='data:image/png;base64,{}'.format(encoded_image))
])

if __name__ == '__main__':
    app.run_server(debug=True)


#%%