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
import dash_bootstrap_components as dbc

import plotly as ply
import plotly.express as px

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import scipy.stats as st
import statistics
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from statsmodels.graphics.gofplots import qqplot

import sys
import os
import datetime
import time

import requests
from bs4 import BeautifulSoup
import re

#import json
#import nltk

print("\nIMPORT SUCCESS")

#%%
# Absolute path of current folder
#abspath_curr = '/content/drive/My Drive/SPORTS/NCAAB/'

# Absolute path of shallow utilities folder
#abspath_util_shallow = '/content/drive/My Drive/Colab Notebooks/teaching/gwu/machine_learning_I/spring_2022/code/utilities/p2_shallow_learning/'

#%%
# CLEAN DATA IMPORT
rollup_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/ncaab_data_rollup_5-2-22'
historical_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/MNCAAB-historical'
tr_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/tr_data_hub_4-05-22'
kp_filepath = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project/data/kenpom_pull_3-14-22'

# NOTE: PANDAS OPENPYXL PACKAGE / EXTENSION REQUIRED TO IMPORT .xlsx FILES

# ROLLED-UP DATA
rollup = pd.read_excel(rollup_filepath + '.xlsx', sheet_name='ROLLUP') #index_col='Team'

# HISTORICAL GAME DATA
regular = pd.read_excel(historical_filepath + '.xlsx', sheet_name='REGULAR') #index_col='Team'
tourney = pd.read_excel(historical_filepath + '.xlsx', sheet_name='TOURNEY') #index_col='Team'

# TEAMRANKINGS DATA
tr = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'

# KENPOM DATA
kp = pd.read_excel(kp_filepath + '.xlsx') #index_col='Team'

print("\nIMPORT SUCCESS")

#%%
# FINAL PRE-PROCESSING
tr['opponent-stocks-per-game'] = tr['opponent-blocks-per-game'] + tr['opponent-steals-per-game']
rollup['opponent-stocks-per-game'] = rollup['opponent-blocks-per-game'] + rollup['opponent-steals-per-game']

tr = tr.round(2)
rollup = rollup.round(2)

print(tr.head())
#print(tr.info())
#print(tr.index)
#print(tr)


#%%
print(regular.columns)
print('*'*100)
print(tourney.columns)

#%% [markdown]
# * Columns are identical across both historical game data sets (Regular Season / Tournament)

#%%
print(regular.info())
print('*'*50)
print(tourney.info())

#%%
# REFINED DATAFRAMES - keeping only unique, essential, or most valuable columns from each data set.

tr_df = tr[['Team', 'win-pct-all-games',
               'average-scoring-margin', #'opponent-average-scoring-margin',
               'points-per-game', 'opponent-points-per-game',
               'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
               'effective-field-goal-pct', 'opponent-effective-field-goal-pct',
               #'true-shooting-percentage',  #'opponent-true-shooting-percentage',
               'three-point-pct', 'two-point-pct', 'free-throw-pct',
               'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
               'assists-per-game', 'opponent-assists-per-game',
               #'turnovers-per-game', 'opponent-turnovers-per-game',
               'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
               'stocks-per-game', 'opponent-stocks-per-game',
               #'blocks-per-game', 'steals-per-game',
               #'opponent-blocks-per-game','opponent-steals-per-game',
               ]]


rollup_df = rollup[['TR_Team', 'win-pct-all-games',
                    'average-scoring-margin', #'opponent-average-scoring-margin',
                    'points-per-game', 'opponent-points-per-game',
                    'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
                    'effective-field-goal-pct', 'opponent-effective-field-goal-pct',
                    #'true-shooting-percentage',  #'opponent-true-shooting-percentage',
                    'three-point-pct', 'two-point-pct', 'free-throw-pct',
                    'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
                    'assists-per-game', 'opponent-assists-per-game',
                    #'turnovers-per-game', 'opponent-turnovers-per-game',
                    'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
                    'stocks-per-game', 'opponent-stocks-per-game',
                    'alias', 'turner_name', 'conf_alias', # 'name', 'school_ncaa',
                    'venue_city', 'venue_state', 'venue_name', 'venue_capacity', #'venue_id', 'GBQ_id',
                    #'logo_large', 'logo_medium', 'logo_small',
                    'mascot', 'mascot_name', 'mascot_common_name', 'tax_species', 'tax_genus', 'tax_family',
                    'tax_order', 'tax_class', 'tax_phylum', 'tax_kingdom', 'tax_domain',
                    'Conference', 'Rank', 'Seed', 'Win', 'Loss',
                    'Adj EM', 'AdjO', 'AdjD', 'AdjT', 'Luck',
                    'SOS Adj EM', 'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
                    ]]


#%%
# RENAME COLUMNS TO IMPROVE APP OPTICS
app_cols = {'Team': 'TEAM', 'win-pct-all-games':'WIN%',
            'average-scoring-margin':'AVG_MARGIN', #'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'points-per-game': 'PTS/GM',  'opponent-points-per-game':'OPP_PTS/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF', 'net-adj-efficiency':'NET_EFF',
            'effective-field-goal-pct':'EFG%', #'true-shooting-percentage':'TS%',
            'opponent-effective-field-goal-pct':'OPP_EFG%', #'opponent-true-shooting-percentage':'OPP_TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%',
            'assists-per-game':'AST/GM', 'opponent-assists-per-game':'OPP_AST/GM',
            'assist--per--turnover-ratio':'AST/TO', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM',
            #'turnovers-per-game':'TO/GM', 'opponent-turnovers-per-game':'OPP_TO/GM',
            #'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM', 'blocks-per-game':'B/GM', 'steals-per-game':'S/GM',
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
            'net-adj-efficiency':'NET_EFF',
            'net-effective-field-goal-pct':'NET_EFG%', 'net-true-shooting-percentage':'NET_TS%',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM', 'total-turnovers-per-game':'TTL_TO/GM',
            'net-assist--per--turnover-ratio':'NET_AST/TO',
            'net-total-rebounds-per-game':'NET_TREB/GM', 'net-off-rebound-pct':'NET_OREB%', 'net-def-rebound-pct':'NET_DREB%'
            }

print(tr_df.info())

#%%
tr_df.columns = tr_df.columns.map(app_cols)

print(tr_df.columns)
print(tr_df.info())

#%%
# DATA ROLLUP - PRE-PROCESSING
print(rollup.info())

#%%
# ROLLUP COLUMNS (FILTERED)
print(rollup_df.columns)
print('*'*100)

# GBQ / KP COLS
#print(f'BIG QUERY / KENPOM DATA:')
#print(rollup_df.columns[63:]) #print(rollup.columns[-40:])

#%%
# DROP NULL VALUES
rollup_df.dropna(inplace=True)
print(rollup_df.info())

#%%
# PRE-PROCESSING ROLLUP FILE
roll_cols = {'TR_Team': 'TEAM', 'win-pct-all-games':'WIN%',
            'average-scoring-margin':'AVG_MARGIN', #'opponent-average-scoring-margin':'OPP_AVG_MARGIN',
            'points-per-game': 'PTS/GM',  'opponent-points-per-game':'OPP_PTS/GM',
            'offensive-efficiency':'O_EFF', 'defensive-efficiency':'D_EFF', 'net-adj-efficiency':'NET_EFF',
            'effective-field-goal-pct':'EFG%', #'true-shooting-percentage':'TS%',
            'opponent-effective-field-goal-pct':'OPP_EFG%', #'opponent-true-shooting-percentage':'OPP_TS%',
            'three-point-pct':'3P%', 'two-point-pct':'2P%', 'free-throw-pct':'FT%',
            'opponent-three-point-pct':'OPP_3P%', 'opponent-two-point-pct':'OPP_2P%', 'opponent-free-throw-pct':'OPP_FT%',
            'assists-per-game':'AST/GM', 'opponent-assists-per-game':'OPP_AST/GM',
            'assist--per--turnover-ratio':'AST/TO', 'opponent-assist--per--turnover-ratio':'OPP_AST/TO',
            'stocks-per-game':'S+B/GM', 'opponent-stocks-per-game':'OPP_S+B/GM',
            #'turnovers-per-game':'TO/GM', 'opponent-turnovers-per-game':'OPP_TO/GM',
            #'opponent-blocks-per-game':'OPP_BLK/GM', 'opponent-steals-per-game':'OPP_STL/GM', 'blocks-per-game':'B/GM', 'steals-per-game':'S/GM',
            'alias':'ABBR', 'name':'NICKNAME', 'turner_name':'INSTITUTION', 'conf_alias':'CONF',
            'venue_city':'CITY', 'venue_state':'STATE', 'venue_name':'ARENA', 'venue_capacity':'ARENA_CAP',
            #'logo_large', 'logo_medium', 'logo_small',
            'mascot':'MASCOT', 'mascot_name':'MASCOT_NAME', 'mascot_common_name':'MASCOT_LABEL',
            'tax_species':'SPECIES', 'tax_genus':'GENUS', 'tax_family':'FAMILY',
            'tax_order':'ORDER', 'tax_class':'CLASS', 'tax_phylum':'PHYLUM', 'tax_kingdom':'KINGDOM', 'tax_domain':'DOMAIN',
            'Conference':'KP_CONF', 'Rank':'KP_RANK', 'Seed':'SEED', 'Win':'WIN', 'Loss':'LOSS', 'Adj EM':'ADJ_EM',
            'AdjO':'ADJ_O', 'AdjD':'ADJ_D', 'AdjT':'ADJ_T', 'Luck':'LUCK',
            'SOS Adj EM':'SOS_ADJ_EM', 'SOS OppO':'SOS_OPP_O', 'SOS OppD':'SOS_OPP_D', 'NCSOS Adj EM':'NCSOS_ADJ_EM'
            }

#%%
# MAP COLUMN LABELING TO DATAFRAME
rollup_df.columns = rollup_df.columns.map(roll_cols)

print(rollup_df.columns)
print(rollup_df.info())


#%%
# GRAPH SCRATCH

def display_chart(stata, statb, statc, statd):
    fig = px.scatter(tr_df, x='WIN%', y=[stata])
    return fig


#fig.update_layout(height=600, width=800, title_text="Side By Side Subplots")
#shared_yaxes=True, horizontal_spacing=0.0025

#fig.show(rendered='browser')

#hover_data=['petal_width', 'petal_length']

#histogram = px.histogram(test, x='Probability', color=TARGET,
#                         marginal="box", nbins=30, opacity=0.6, range_x = [-5, 5]
#                         color_discrete_sequence=['#FFBD59',
#                                                  '#3BA27A'])

#%%




#%%

#%% [markdown]
# * NORMALITY TESTS
# *

#%%
print(rollup_df.columns)
print(rollup_df.info())


#%%
rollup_df.index = rollup_df['TEAM']
rollup_df.drop(columns='TEAM', inplace=True)

#%%
float_rollup = rollup_df[['WIN%', 'AVG_MARGIN', 'PTS/GM', 'OPP_PTS/GM', 'O_EFF', 'D_EFF',
                          'NET_EFF', 'EFG%', 'OPP_EFG%', '3P%', '2P%', 'FT%',
                          'OPP_3P%', 'OPP_2P%', 'OPP_FT%', 'AST/GM', 'OPP_AST/GM', 'AST/TO', 'OPP_AST/TO',
                          'S+B/GM', 'OPP_S+B/GM',
                          'ARENA_CAP', 'KP_RANK', 'SEED', 'WIN', 'LOSS',
                          'ADJ_EM', 'ADJ_O', 'ADJ_D', 'ADJ_T', 'LUCK',
                          'SOS_ADJ_EM', 'SOS_OPP_O', 'SOS_OPP_D', 'NCSOS_ADJ_EM']]

print(float_rollup.info())

#%%
#test = st.kstest(float_rollup['EFG%'], 'norm')
da_testtv = st.normaltest(float_rollup['EFG%'])

print(da_testtv)

#%%
## NORMALITY TESTS
normality_df = pd.DataFrame()
#for col in float_rollup.columns:

normality_df['WIN%_KS'] = st.kstest(float_rollup['WIN%'], 'norm')

    #kstest_t = st.kstest(df['temp'], 'norm')
    #da_testtv = st.normaltest(df['traffic_volume'])
    #da_testt = st.normaltest(df['temp'])
    #shapiro_testtv = st.shapiro(df['traffic_volume'])
    #shapiro_testt = st.shapiro(df['temp'])

print(normality_df)
#%%
## NORMALITY TESTS
kstest_tv = st.kstest(df['traffic_volume'],'norm')
kstest_t = st.kstest(df['temp'],'norm')
da_testtv = st.normaltest(df['traffic_volume'])
da_testt = st.normaltest(df['temp'])
shapiro_testtv = st.shapiro(df['traffic_volume'])
shapiro_testt = st.shapiro(df['temp'])

#%%
# NUMERICS
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numeric_cols = rollup_df.select_dtypes(include=numerics)
#numeric_cols.dropna(inplace=True)

X = rollup_df[rollup_df._get_numeric_data().columns.to_list()[:-1]]
Y = rollup_df['WIN%']
#X.drop(columns='price', inplace=True, axis=1)

#%%
print(X.info())

#%%
X = StandardScaler().fit_transform(X)

#%%

pca = PCA(n_components=10, svd_solver='full') # 'mle'

pca.fit(X)
X_PCA = pca.transform(X)
print('ORIGINAL DIMENSIONS:', X.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')


#%%
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.xticks(x)
#plt.grid()
plt.show()

#%% [markdown]

# 10 features explain ~89.6% of variance


#%%
# SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD]
# CONDITION NUMBER

# ORIGINAL DATA

from numpy import linalg as LA

H = np.matmul(X.T, X)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES {d}')
print(f'ORIGINAL DATA: CONDITIONAL NUMBER {LA.cond(X)}')


#%%
# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES {d_PCA}')
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER {LA.cond(X_PCA)}')
print('*'*58)

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET

#pca_df = pca.explained_variance_ratio_

a, b = X_PCA.shape
column = []
df_pca = pd.DataFrame(X_PCA).corr()

for i in range(b):
    column.append(f'PRINCIPLE COLUMN {i+1}')
sns.heatmap(df_pca, annot=True, xticklabels=column, yticklabels=column)
df_PCA = pd.DataFrame(data=X_PCA)
plt.title("correlation coefficient")
plt.show()

df_PCA = pd.concat([df_PCA, Y], axis=1)
df_PCA.columns = pd.DataFrame(data=df, columns=col)

df_PCA.info()

#print("old one:",df1.head().to_string)
#print("new one:",df_PCA.head().to_string)

#%% [markdown]

# SINGULAR VALUE DECOMPOSITION [SVD]
# * Used in tandem with Principal Component Analysis
#     * Dimensionality Reduction - reducing number of input variables / features
# * Create projection of sparse dataset prior to fitting a model
#     * Always outputs a square matrix H (transposed X)
#     * Singular values close to zero to be removed
# * Singular Value Decomposition
#     * Higher = GOOD
# * Conditional Number
#    * Higher = BAD
# * Eigensystem Analysis
#     * Eigenvalues
#    * Eigenvectors
#     * Condition Number - max value / min number
#         * LA.cond(X)
#         * <100 = weak multicollinearity
#         * 100<k<1000 = moderate multicollinearity
#         * >1000 = severe multicollinearity



#%% [markdown]
## VISUALIZATION INDEX:
# 1) Line-plot
# 2) Bar-plot (stacked, grouped)
# 3) Count-plot
# 4) Cat-plot
# 5) Pie-chart
# 6) Dis-plot
# 7) Pair plot
# 8) Heatmap
# 9) Hist-plot
# 10) QQ-plot
# 11) Kernel density estimate
# 12) Scatter plot and regression line (sklearn)
# 13) Multivariate Box plot

# COLUMNS
# * ['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
#        'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
#        'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
#        'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']

# CONFERENCES
# A10
# AAC
# ACC
# AE
# AS
# BIG10
# BIG12
# BIGEAST
# BIGSKY
# BIGSOUTH
# BIGWEST
# COLONIAL
# CUSA
# HORIZON
# IVY
# MAAC
# MAC
# MEAC
# MVC
# MWC
# NE
# OVC
# PAC12
# PATRIOT
# SEC
# SOUTHERN
# SOUTHLAND
# SUMMIT
# SUNBELT
# SWAC
# WAC
# WCC

#%%
# VARIABLES
print(regular.index)
print(regular['WTeamID'])

#%%
# 1) Line-plot
plt.figure(figsize=(8,6))
sns.lineplot(data=regular, x='Season', y='WFGA3', palette='magma', markers=True)
plt.title('WINNER 3PT ATTEMPTS BY SEASON', fontsize=16)
plt.xlabel('SEASON', fontsize=16)
plt.ylabel('WINNER 3PT ATTEMPTS', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

#for i in tr_df.index:
  #plt.text(tr_df['offensive-efficiency'][tr_data_hub.index==i]+.01,tr_data_hub['defensive-efficiency'][tr_data_hub.index==i]+.01,str(i), color='black')

plt.show()


#%%
# 2) Bar-plot (stacked, grouped)


#%%
# 3) Count-plot


#%%
# 4) Cat-plot


#%%
# 5) Pie-chart


#%%
# 6) Dis-plot


#%%
# 7) Pair plot


#%%
# 8) Heatmap


#%%
# 9) Hist-plot


#%%
# 10) QQ-plot


#%%
# 11) Kernel density estimate


#%%
# 12) Scatter plot and regression line (sklearn)
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
# 13) Multivariate Box plot


#%%

14)	 Area plot (if applicable)

qqplot(df['traffic_volume'])
plt.title("QQ-plot of traffic volume ")
plt.show()
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

table = go.Figure(data=[go.Table(
    header=dict(values=columns, fill_color='#FFBD59',
                line_color='white', align='center',
                font=dict(color='white', size=13)),
    cells=dict(values=[test[c] for c in columns],
               format=["d", "", "", "", "", ".2%"],
               fill_color=[['white', '#FFF2CC']*(len(test)-1)],
               align='center'))
])
table.update_layout(title_text=f'Sample records (n={len(test)})',
                    font_family='Tahoma')

#%%

html.H3('Slider 1'), \
dcc.Slider(id='slider-1',
           min=0,
           max=20,
           value=10)

html.H3('Slider 2'),
dcc.Slider(id='slider-2',
           min=0,
           max=20,
           value=10)

html.H3('Slider 3'),
dcc.Slider(id='slider-3',
           min=0,
           max=20,
           value=10)

html.H3('Slider 4'),
dcc.Slider(id='slider-4',
           min=0,
           max=20,
           value=10)

#%%