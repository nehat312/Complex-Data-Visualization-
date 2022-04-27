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
# TEAM STATS (TEAM RANKINGS)

## URL VARIABLES

# filtered stats of interest

title_links = ['points-per-game', 'average-scoring-margin', 'field-goals-made-per-game',
               'field-goals-attempted-per-game', 'offensive-efficiency', 'defensive-efficiency', 'effective-possession-ratio',
               'effective-field-goal-pct', 'true-shooting-percentage', 'three-point-pct', 'two-point-pct',
               'free-throw-pct', 'three-pointers-made-per-game', 'three-pointers-attempted-per-game',
               'offensive-rebounds-per-game', 'defensive-rebounds-per-game', 'total-rebounds-per-game',
               'offensive-rebounding-pct', 'defensive-rebounding-pct', 'total-rebounding-percentage',
               'blocks-per-game', 'steals-per-game', 'assists-per-game', 'turnovers-per-game',
               'assist--per--turnover-ratio', 'win-pct-all-games', 'win-pct-close-games', 'possessions-per-game',
               'personal-fouls-per-game',
               'opponent-points-per-game', 'opponent-average-scoring-margin', 'opponent-shooting-pct',
               'opponent-effective-field-goal-pct', 'opponent-true-shooting-percentage',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'opponent-assists-per-game', 'opponent-turnovers-per-game', 'opponent-assist--per--turnover-ratio',
               'opponent-offensive-rebounds-per-game', 'opponent-defensive-rebounds-per-game',
               'opponent-total-rebounds-per-game', 'opponent-offensive-rebounding-pct', 'opponent-defensive-rebounding-pct',
               'opponent-blocks-per-game', 'opponent-steals-per-game', 'opponent-effective-possession-ratio',
               ]

team_links = ['points-per-game', 'average-scoring-margin',
              'offensive-efficiency', 'percent-of-points-from-2-pointers',
              'percent-of-points-from-3-pointers', 'percent-of-points-from-free-throws',
              'shooting-pct', 'effective-field-goal-pct', 'true-shooting-percentage',
              'three-point-pct', 'two-point-pct', 'free-throw-pct',
              'field-goals-made-per-game', 'field-goals-attempted-per-game',
              'three-pointers-made-per-game', 'three-pointers-attempted-per-game',
              'free-throws-made-per-game', 'free-throws-attempted-per-game',
              'three-point-rate', 'fta-per-fga', 'ftm-per-100-possessions',
              'offensive-rebounds-per-game', 'defensive-rebounds-per-game',
              'total-rebounds-per-game',
              'offensive-rebounding-pct', 'defensive-rebounding-pct',
              'total-rebounding-percentage', 'blocks-per-game',
              'steals-per-game', 'assists-per-game',
              'turnovers-per-game', 'assist--per--turnover-ratio',
              'assists-per-fgm', 'games-played',
              'possessions-per-game', 'extra-chances-per-game',
              'effective-possession-ratio',
              'win-pct-all-games', 'win-pct-close-games', ]

opponent_links = ['personal-fouls-per-game'
                  'opponent-points-per-game', 'opponent-average-scoring-margin',
                  'defensive-efficiency', 'opponent-points-from-2-pointers',
                  'opponent-points-from-3-pointers', 'opponent-percent-of-points-from-2-pointers',
                  'opponent-percent-of-points-from-3-pointers', 'opponent-percent-of-points-from-free-throws',
                  'opponent-shooting-pct', 'opponent-effective-field-goal-pct',
                  'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
                  'opponent-true-shooting-percentage',
                  'opponent-field-goals-made-per-game', 'opponent-field-goals-attempted-per-game',
                  'opponent-three-pointers-made-per-game',
                  'opponent-three-pointers-attempted-per-game', 'opponent-free-throws-made-per-game',
                  'opponent-free-throws-attempted-per-game',
                  'opponent-three-point-rate', 'opponent-two-point-rate', 'opponent-fta-per-fga',
                  'opponent-ftm-per-100-possessions',
                  'opponent-free-throw-rate', 'opponent-non-blocked-2-pt-pct',
                  'opponent-offensive-rebounds-per-game', 'opponent-defensive-rebounds-per-game',
                  'opponent-team-rebounds-per-game', 'opponent-total-rebounds-per-game',
                  'opponent-offensive-rebounding-pct', 'opponent-defensive-rebounding-pct',
                  'opponent-blocks-per-game', 'opponent-steals-per-game', 'opponent-block-pct',
                  'opponent-steals-perpossession',
                  'opponent-steal-pct', 'opponent-assists-per-game', 'opponent-turnovers-per-game',
                  'opponent-assist--per--turnover-ratio',
                  'opponent-assists-per-fgm', 'opponent-assists-per-possession', 'opponent-turnovers-per-possession',
                  'opponent-turnover-pct', 'opponent-personal-fouls-per-game',
                  'opponent-personal-fouls-per-possession', 'opponent-personal-foul-pct',
                  'opponent-effective-possession-ratio',
                  'opponent-win-pct-all-games', 'opponent-win-pct-close-games']

#%%
## TEAMRANKINGS.COM - DATA SCRAPE

tr_url = 'https://www.teamrankings.com/ncaa-basketball/stat/'
base_url = 'https://www.teamrankings.com/'

tr_cols = ['Rank', 'Team', '2021', 'Last 3', 'Last 1', 'Home', 'Away', '2020']  # , 'Stat'
tr_link_dict = {link: pd.DataFrame() for link in title_links}  # columns=tr_cols
df = pd.DataFrame()

for link in title_links:
    stat_page = requests.get(tr_url + link)
    soup = BeautifulSoup(stat_page.text, 'html.parser')
    table = soup.find_all('table')[0]
    cols = [each.text for each in table.find_all('th')]
    rows = table.find_all('tr')
    for row in rows:
        data = [each.text for each in row.find_all('td')]
        temp_df = pd.DataFrame([data])
        # df = df.append(temp_df, sort=True).reset_index(drop=True)
        tr_link_dict[link] = tr_link_dict[link].append(temp_df, sort=True).reset_index(drop=True)
        tr_link_dict[link] = tr_link_dict[link].dropna()

    tr_link_dict[link].columns = cols
    tr_link_dict[link][link] = tr_link_dict[link]['2021']
    tr_link_dict[link].index = tr_link_dict[link]['Team']
    tr_link_dict[link].drop(columns=['Rank', 'Last 3', 'Last 1', 'Home', 'Away', '2020', '2021', 'Team'], inplace=True)

print(tr_link_dict.keys())


#%%
tr_df = pd.DataFrame()

for stat in tr_link_dict:
    # tr_link_dict[stat].replace({'%',''}, regex=True)#.strip('%')
    tr_df[stat] = tr_link_dict[stat]
    # tr_link_dict[stat] = float(tr_link_dict[stat].replace('%',''))

objects = tr_df.select_dtypes(['object'])
tr_df[objects.columns] = objects.apply(lambda x: x.str.strip('%'))

for stat in tr_df:
    tr_df[stat] = pd.to_numeric(tr_df[stat])

# for col in tr_df:
# tr_df[col] = tr_df[col].astype(float)
# pd.to_numeric(df['DataFrame Column'],errors='coerce')

tr_df.head()

# tr_df[stat] = tr_df[stat].replace('%','') #, regex=True
# pd.DataFrame.from_dict(tr_link_dict.keys())
# tr# tr_df[stat] = tr_df[stat].replace('%','') #, regex=True
# print(tr_link_dict['two-point-pct'])

print(tr_df.describe())

#%%
## RAW DATA EXPORT

tr_filepath_raw = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_ncaab_data_4-05-22-raw'

tr_df.to_excel(tr_filepath_raw + '.xlsx', index=True)
tr_df.to_csv(tr_filepath_raw + '.csv', index=True)

print("\nEXPORT SUCCESS")

#%%
## RAW DATA IMPORT

tr_filepath_raw = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_ncaab_data_4-05-22-raw'

tr_df = pd.read_excel(tr_filepath_raw + '.xlsx', index_col='Team')
# tr_df = pd.read_csv(tr_filepath_raw + '.csv', index_col='Team')

print("\nIMPORT SUCCESS")

#%%
## FEATURE ENGINEERING

tr_df.info()

#%%
# SCORING MARGIN / POSSESSIONS
tr_df['net-avg-scoring-margin'] = tr_df['average-scoring-margin'] - tr_df['opponent-average-scoring-margin']
tr_df['net-points-per-game'] = tr_df['points-per-game'] - tr_df['opponent-points-per-game']
#tr_df['net-effective-possession-ratio'] = tr_df['effective-possession-ratio'] - tr_df['opponent-effective-possession-ratio']
tr_df['net-adj-efficiency'] = tr_df['offensive-efficiency'] - tr_df['defensive-efficiency']

# NET SHOOTING PERCENTAGES
tr_df['net-effective-field-goal-pct'] = tr_df['effective-field-goal-pct'] - tr_df['opponent-effective-field-goal-pct']
tr_df['net-true-shooting-percentage'] = tr_df['true-shooting-percentage'] - tr_df['opponent-true-shooting-percentage']

# STOCKS = STEALS + BLOCKS
tr_df['stocks-per-game'] = tr_df['steals-per-game'] + tr_df['blocks-per-game']
#tr_df['opponent-stocks-per-game'] = tr_df['opponent-steals-per-game'] + tr_df['opponent-blocks-per-game']
#tr_df['net-stocks-per-game'] = tr_df['stocks-per-game'] - tr_df['opponent-stocks-per-game']

# AST/TO = TURNOVERS / ASSISTS
tr_df['total-turnovers-per-game'] = tr_df['turnovers-per-game'] + tr_df['opponent-turnovers-per-game']
tr_df['net-assist--per--turnover-ratio'] = tr_df['assist--per--turnover-ratio'] - tr_df[
    'opponent-assist--per--turnover-ratio']

# REBOUNDS
tr_df['net-total-rebounds-per-game'] = tr_df['total-rebounds-per-game'] - tr_df['opponent-total-rebounds-per-game']
tr_df['net-off-rebound-pct'] = tr_df['offensive-rebounding-pct'] - tr_df['opponent-offensive-rebounding-pct']
tr_df['net-def-rebound-pct'] = tr_df['defensive-rebounding-pct'] - tr_df['opponent-defensive-rebounding-pct']

# ALTERNATE CALC - yields different performance than above
tr_df['net-off-rebound-pct'] = tr_df['offensive-rebounding-pct'] - tr_df['opponent-defensive-rebounding-pct']
tr_df['net-def-rebound-pct'] = tr_df['defensive-rebounding-pct'] - tr_df['opponent-offensive-rebounding-pct']

tr_df.info()
# tr_df.columns


#%%
## FINAL DATA EXPORT

tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'

tr_df.to_excel(tr_filepath + '.xlsx', index=True)
tr_df.to_csv(tr_filepath + '.csv', index=True)

print("\nEXPORT SUCCESS")


#%%
# LIVE MATCHUPS
tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
kp_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
tr_df = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'
#tr_df = pd.read_csv(mtr_filepath + '.csv')
kp_df = pd.read_excel(kp_filepath + '.xlsx') #index_col='Team'
#kp_df = pd.read_csv(kp_filepath + '.csv')

#print(matchup_df.columns)
print(tr_df.head())

#%%
print(tr_df.info())
print(kp_df.info())
#print(tr_df.index)
#print(tr_df)

#%%

cols_original = ['points-per-game', 'average-scoring-margin', 'field-goals-made-per-game',
               'field-goals-attempted-per-game',
               'offensive-efficiency', 'defensive-efficiency',
               'effective-field-goal-pct', 'true-shooting-percentage', 'three-point-pct', 'two-point-pct',
               'free-throw-pct',
               'three-pointers-made-per-game', 'three-pointers-attempted-per-game',
               'offensive-rebounds-per-game', 'defensive-rebounds-per-game', 'total-rebounds-per-game',
               'offensive-rebounding-pct', 'defensive-rebounding-pct', 'total-rebounding-percentage',
               'blocks-per-game', 'steals-per-game', 'assists-per-game', 'turnovers-per-game',
               'assist--per--turnover-ratio', 'win-pct-all-games', 'win-pct-close-games', 'possessions-per-game',
               'personal-fouls-per-game', 'opponent-points-per-game', 'opponent-average-scoring-margin', 'opponent-shooting-pct',
               'opponent-effective-field-goal-pct',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'opponent-true-shooting-percentage',
               'opponent-assists-per-game', 'opponent-turnovers-per-game', 'opponent-assist--per--turnover-ratio',
               ]

cols_app = {'points-per-game':'PPG', 'average-scoring-margin':'AVG_MARGIN', 'win-pct-all-games':'WIN%', 'win-pct-close-games':'WIN%_CLOSE',
            'effective-field-goal-pct':'EFG%', 'true-shooting-percentage':'TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'field-goals-made-per-game':'FGM/GM', 'field-goals-attempted-per-game':'FGA/GM', 'three-pointers-made-per-game':'3PM/GM', 'three-pointers-attempted-per-game':'3PA/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF',
            'total-rebounds-per-game':'TRB/GM', 'offensive-rebounds-per-game':'ORB/GM', 'defensive-rebounds-per-game':'DRB/GM',
            'offensive-rebounding-pct':'PPG', 'defensive-rebounding-pct':'PPG', 'total-rebounding-percentage':'PPG',
            'blocks-per-game':'B/GM', 'steals-per-game':'S/GM', 'assists-per-game':'PPG', 'turnovers-per-game':'PPG',
            'assist--per--turnover-ratio':'AST/TO', 'possessions-per-game':'POSS/GM', 'personal-fouls-per-game':'PF/GM',
            'opponent-points-per-game':'OPP_PPG', 'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'opponent-effective-field-goal-pct':'OPP_EFG%', 'opponent-true-shooting-percentage':'OPP_TS%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%', 'opponent-shooting-pct':'OPP_FG%',
            'opponent-assists-per-game':'OPP_AST/GM', 'opponent-turnovers-per-game':'OPP_TO/GM', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            }

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
tr_df.columns = tr_df.map(cols_app)
tr_df.info()

#%%

print(tourney_teams.columns)
print(tourney_teams_dict.keys())
print(tourney_teams_dict.values())
print(tourney_teams_dict)
#%%
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
ncaab_app = dash.Dash('NCAAM BASKETBALL DASHBOARD', external_stylesheets=external_stylesheets)
application = ncaab_app.server

# {current_time:%Y-%m-%d %H:%M}
#'backgroundColor': 'rgb(220, 220, 220)',

ncaab_app.layout = html.Div([html.H1('NBA MATCHUP MACHINE', style={'textAlign': 'Center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}),
                     #html.Br(),
                               html.Div([html.H3((f'MATCHUP INFORMATION - 4/7/2022'),
                                                 style={'textAlign': 'center', 'backgroundColor': 'rgb(223,187,133)', 'color': 'black', 'fontWeight': 'bold', 'border': '3px solid black'}), #'dodgerblue'

                               dash_table.DataTable(tr_df.to_dict('records'), [{"name": i, "id": i} for i in tr_df.columns],
                                                    style_data={'textAlign': 'Center', 'fontWeight': 'bold', 'border': '2px solid black'},
                                                    style_cell={'textAlign': 'Center', 'fontWeight': 'bold', 'padding': '5px'},   #324f6e - TBD  #B10DC9 - fuschia #7FDBFF - Aqua
                                                    style_header={'backgroundColor': '#7FDBFF', 'color': 'black', 'fontWeight': 'bold', 'border': '2px solid black'}, #'1px solid blue'
                                                    style_data_conditional = [
                                                                              #{'if': {'row_index': [2,3,6,7,10,11]}, 'backgroundColor': 'rgb(211, 211, 211)'},
                                                                              {'if': {'filter_query': '{FINAL} > 105', 'column_id': 'FINAL', 'row_index': [0,1]},
                                                                               'color': '#2ECC40', 'fontWeight': 'bold'}, #'backgroundColor': 'red':'#2ECC40', gray?: '#FF4136'


                                                                              {'if': {'filter_query': '{MODEL_PREDICTION} = "LOSS"','column_id': 'MODEL_PREDICTION'},
                                                                               'color': 'black', 'backgroundColor': '#FF4136', 'fontWeight': 'bold'}, # RED '#FF4136' / GREEN '#2ECC40'

                                                                              {'if': {'column_id': 'MODEL_PREDICTION'}, 'fontWeight': 'bold',},], #'#01FF70'  'backgroundColor': '#01FF70',
                                                    ),
                               ]),
                               html.Br(),
                               #html.P('METRIC COMPARISON'),
                               dcc.Graph(id='matchup-chart'),
                               html.Br(),
                               html.P('STAT A'),
                               dcc.Dropdown(id='stata',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     {'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     {'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     {'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     {'label': 'TREB%', 'value': 'TREB%'},
                                                     {'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     {'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},], value='WIN%'),
                               html.Br(),
                               html.P('STAT B'),
                               dcc.Dropdown(id='statb',
                                            options=[{'label': 'WIN%', 'value': 'WIN%'},
                                                     {'label': 'AVG_MARGIN', 'value': 'AVG_MARGIN'},
                                                     {'label': 'OFF_EFF', 'value': 'OFF_EFF'},
                                                     {'label': 'DEF_EFF', 'value': 'DEF_EFF'},
                                                     {'label': 'EFG%', 'value': 'EFG%'},
                                                     {'label': 'TS%', 'value': 'TS%'},
                                                     {'label': 'OPP_EFG%', 'value': 'OPP_EFG%'},
                                                     {'label': 'OPP_TS%', 'value': 'OPP_TS%'},
                                                     {'label': 'AST/TO', 'value': 'AST/TO'},
                                                     {'label': 'OPP_AST/TO', 'value': 'OPP_AST/TO'},
                                                     {'label': 'TREB%', 'value': 'TREB%'},
                                                     {'label': 'STL+BLK/GM', 'value': 'STL+BLK/GM'},
                                                     {'label': 'OPP_STL+BLK/GM', 'value': 'OPP_STL+BLK/GM'},], value='WIN%'),
                               html.Br(),

                               html.Div(id='layout')])

@ncaab_app.callback(Output(component_id='layout', component_property='children'),
                      #Output(component_id='matchup-df', component_property='figure'),
                      Output(component_id='matchup-chart', component_property='figure'),
                      [Input(component_id='stata', component_property='value'),
                       Input(component_id='statb', component_property='value')])

def display_chart(stata, statb):
    fig = px.scatter(ncaab_app, x=[stata], y=[statb])
    return fig

if __name__ == '__main__':
    application.run(host='0.0.0.0', port=8050)


#%%


#%%

# SCATTERPLOT -
plt.figure(figsize=(10,10))
sns.scatterplot(data=tr_data_hub, x='offensive-efficiency', y='defensive-efficiency', palette='magma', markers=True)
plt.title('OFF. VS. DEF. EFFICIENCY (BY CONFERENCE)', fontsize=16)
plt.xlabel('OFFENSIVE EFFICIENCY', fontsize=16)
plt.ylabel('DEFENSIVE EFFICIENCY', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

for i in tr_data_hub.index:
  plt.text(tr_data_hub['offensive-efficiency'][tr_data_hub.index==i]+.01,tr_data_hub['defensive-efficiency'][tr_data_hub.index==i]+.01,str(i), color='black')

plt.show()


#%%

## KENPOM

# KENPOM

## DATA SCRAPE

# KENPOM DATA SCRAPE

# Base url, and a lambda func to return url for a given year
base_url = 'http://kenpom.com/index.php'
url_year = lambda x: '%s?y=%s' % (base_url, str(x) if x != 2021 else base_url)

# Years on kenpom's site; scrape and set as list to be more dynamic?
years = range(2021, 2022)


# Create a method that parses a given year and spits out a raw dataframe
def import_raw_year(year):
    """
    Imports raw data from a kenpom year into a dataframe
    """
    f = requests.get(url_year(year))
    soup = BeautifulSoup(f.text, "lxml")
    table_html = soup.find_all('table', {'id': 'ratings-table'})

    # Weird issue w/ <thead> in the html
    # Prevents us from just using pd.read_html
    # Find all <thead> contents and replace/remove them
    # This allows us to easily put the table row data into a dataframe using pandas
    thead = table_html[0].find_all('thead')

    table = table_html[0]
    for x in thead:
        table = str(table).replace(str(x), '')

    kp_df = pd.read_html(table)[0]
    kp_df['year'] = year
    return kp_df


# Import all the years into a singular dataframe
kp_df = None
for x in years:
    kp_df = pd.concat((kp_df, import_raw_year(x)), axis=0) if kp_df is not None else import_raw_year(2022)

# Column rename based off of original website
kp_df.columns = ['Rank', 'Team', 'Conference', 'W-L', 'Adj EM',
                 'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank',
                 'AdjT', 'AdjT Rank', 'Luck', 'Luck Rank',
                 'SOS Adj EM', 'SOS Adj EM Rank', 'SOS OppO', 'SOS OppO Rank',
                 'SOS OppD', 'SOS OppD Rank', 'NCSOS Adj EM', 'NCSOS Adj EM Rank', 'Year']

# Lambda that returns true if given string is a number and a valid seed number (1-16)
valid_seed = lambda x: True if str(x).replace(' ', '').isdigit() \
                               and int(x) > 0 and int(x) <= 16 else False

# Use lambda to parse out seed/team
kp_df['Seed'] = kp_df['Team'].apply(lambda x: x[-2:].replace(' ', '') \
    if valid_seed(x[-2:]) else np.nan)

kp_df['Team'] = kp_df['Team'].apply(lambda x: x[:-2] if valid_seed(x[-2:]) else x)

# Split W-L column into Win / Loss
kp_df['Win'] = kp_df['W-L'].apply(lambda x: int(re.sub('-.*', '', x)))
kp_df['Loss'] = kp_df['W-L'].apply(lambda x: int(re.sub('.*-', '', x)))
kp_df.drop('W-L', inplace=True, axis=1)

# Reorder columns
kp_df = kp_df[['Year', 'Rank', 'Team', 'Conference', 'Win', 'Loss', 'Seed', 'Adj EM',
               'AdjO', 'AdjO Rank', 'AdjD', 'AdjD Rank',
               'AdjT', 'AdjT Rank', 'Luck', 'Luck Rank',
               'SOS Adj EM', 'SOS Adj EM Rank', 'SOS OppO', 'SOS OppO Rank',
               'SOS OppD', 'SOS OppD Rank', 'NCSOS Adj EM', 'NCSOS Adj EM Rank']]

kp_df.info()

## DATA EXPORT

kp_df.to_csv('drive/My Drive/SPORTS/kenpom_pull_3-14-22.csv', index=False)
kp_df.to_excel('drive/My Drive/SPORTS/kenpom_pull_3-14-22.xlsx', index=False)

# Derive the id from the google drive shareable link.

##For the file at hand the link is as below
# URL = 'https://drive.google.com/file/d/1m0mAGzpeMR0W-BDL5BtKrs0HOZsPIAbX/view?usp=sharing'
# path = 'https://drive.google.com/uc?export=download&id='+URL.split('/')[-2]
# df = pd.read_pickle(path)
# df = pd.read_csv(path)


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

# CORRELATION MATRIX
mm_2022.corr()

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
