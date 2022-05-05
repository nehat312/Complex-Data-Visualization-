#%%
## LIBRARY IMPORT
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

print("\nIMPORT SUCCESS")

#%%
## DATA IMPORT
# Set absolute path of current folder
abspath_curr = '/Users/nehat312/GitHub/Complex-Data-Visualization-/project'

#%%
# DATA FILEPATHS
# NOTE: PANDAS OPENPYXL EXTENSION REQUIRED TO IMPORT .xlsx FILES
historical_filepath = abspath_curr + '/data/MNCAAB-historical'
rollup_filepath = abspath_curr + '/data/ncaab_data_rollup_5-2-22'

#tr_filepath = abspath_curr + '/data/tr_data_hub_4-05-22'
#kp_2022_filepath = abspath_curr + '/data/kenpom_pull_3-14-22'

# IMPORT HISTORICAL GAME DATA
regular = pd.read_excel(historical_filepath + '.xlsx', sheet_name='REGULAR') #index_col='Team'
tourney = pd.read_excel(historical_filepath + '.xlsx', sheet_name='TOURNEY') #index_col='Team'

# IMPORT ROLLED-UP DATA
rollup = pd.read_excel(rollup_filepath + '.xlsx', sheet_name='ROLLUP')

# IMPORT TEAMRANKINGS DATA
#tr = pd.read_excel(tr_filepath + '.xlsx') #index_col='Team'

# IMPORT KENPOM DATA
#kp_2022 = pd.read_excel(kp_2022_filepath + '.xlsx') #index_col='Team'

print("\nIMPORT SUCCESS")

#%%
## EXPLORE DATA
print(regular.info())
print('*'*50)
print(tourney.info())
print('*'*50)
print(rollup.info())

#%%
# FEATURE ENGINEERING
regular['WFG%'] = regular['WFGM'] / regular['WFGA']
regular['LFG%'] = regular['LFGM'] / regular['LFGA']
regular['WMargin'] = regular['WScore'] - regular['LScore']

tourney['WFG%'] = tourney['WFGM'] / tourney['WFGA']
tourney['LFG%'] = tourney['LFGM'] / tourney['LFGA']
tourney['WMargin'] = tourney['WScore'] - tourney['LScore']

rollup['opponent-stocks-per-game'] = rollup['opponent-blocks-per-game'] + rollup['opponent-steals-per-game']
rollup = rollup.round(2)

print('REGULAR COLUMNS:')
print(regular.columns)
print('*'*100)
print('TOURNEY COLUMNS:')
print(tourney.columns)
print('*'*100)
print('ROLLUP COLUMNS:')
print(rollup.columns)


#%% [markdown]
# * Columns are identical across both historical game data sets (Regular Season / Tournament)
# * Elements of multi-collinearity exist throughout the data; PCA will be important towards feature selection
# * Feature selection will be critical: keeping only unique, essential, or otherwise valuable columns from each data set.

#%%
## FEATURE SELECTION
rollup_df = rollup[['TR_Team', 'win-pct-all-games', 'average-scoring-margin', #'opponent-average-scoring-margin',
                    'points-per-game', 'opponent-points-per-game',
                    'offensive-efficiency', 'defensive-efficiency', 'net-adj-efficiency',
                    'effective-field-goal-pct', 'opponent-effective-field-goal-pct',
                    'three-point-pct', 'two-point-pct', 'free-throw-pct',
                    'opponent-three-point-pct', 'opponent-two-point-pct', 'opponent-free-throw-pct',
                    'assists-per-game', 'opponent-assists-per-game',
                    'assist--per--turnover-ratio', 'opponent-assist--per--turnover-ratio',
                    'stocks-per-game', 'opponent-stocks-per-game',
                    'alias', 'turner_name', 'conf_alias', # 'name', 'school_ncaa',
                    'venue_city', 'venue_state', 'venue_name', 'venue_capacity', #'venue_id', 'GBQ_id',
                    'mascot', 'mascot_name', 'mascot_common_name', 'tax_species', 'tax_genus', 'tax_family',
                    'tax_order', 'tax_class', 'tax_phylum', 'tax_kingdom', 'tax_domain',
                    'Conference', 'Rank', 'Seed', 'Win', 'Loss',
                    'Adj EM', 'AdjO', 'AdjD', 'AdjT', 'Luck',
                    'SOS Adj EM', 'SOS OppO', 'SOS OppD', 'NCSOS Adj EM'
                    #'logo_large', 'logo_medium', 'logo_small',
                    #'true-shooting-percentage',  #'opponent-true-shooting-percentage',
                    #'turnovers-per-game', 'opponent-turnovers-per-game',
                    ]]

rollup_col_dict = {'TR_Team': 'TEAM', 'win-pct-all-games':'WIN%',
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
rollup_df.columns = rollup_df.columns.map(rollup_col_dict)

print(rollup_df.columns)
print('*'*100)
print(rollup_df.info())

#%%
# FILTERING FOR MARCH MADNESS PARTICIPANTS / RESET INDEX >> TEAM NAME

# HISTORICAL MARCH MADNESS PARTICIPANTS
regular_mm = regular[regular['Seed'] >= 1]
#regular_mm = regular_mm.set_index('TR_Team')

# 2022 MARCH MADNESS (68 TEAMS)
    #regular_mm_2022 = regular[regular['Seed_2022'] >= 1]
    #mm_2022 = rollup_df[rollup_df['Seed'] >= 1]

#%%
## CHECKING FOR NULL / EMPTY VALUES
print('REGULAR NULL VALUES:')
print(regular.isnull().sum().sort_values(ascending=False))
print('*'*50)
print('TOURNEY NULL VALUES:')
print(tourney.isnull().sum().sort_values(ascending=False))
print('*'*50)
print('ROLLUP NULL VALUES:')
print(rollup_df.isnull().sum().sort_values(ascending=False))

#%%
## DROPPING NULL VALUES
regular.dropna(inplace=True)
tourney.dropna(inplace=True)
rollup_df.dropna(inplace=True)

#%%
## EXPLORE DATA
print(regular.info())
print('*'*50)
print(tourney.info())
print('*'*50)
print(rollup_df.info())

#%% [markdown]
# FEATURE IMPORTANCE
# Examining distribution of data with various tests of normality:
    # Normality Tests
        # * K-SQUARED (KS)
        # * D`AGOSTINO-PEARSON (DA)
        # * SHAPIRO-WILK (SHAP)
# Tools / packages below implemented with intent of highlighting features of greatest impact or effect:
    # StandardScaler (SS)
    # Principal Component Analysis (PCA)
    # Singular Value Decomposition (SVD)

#%%
# FILTER FOR NUMERIC COLUMNS
float_rollup = rollup_df[['WIN%', 'AVG_MARGIN', 'PTS/GM', 'OPP_PTS/GM', 'O_EFF', 'D_EFF',
                          'NET_EFF', 'EFG%', 'OPP_EFG%', '3P%', '2P%', 'FT%',
                          'OPP_3P%', 'OPP_2P%', 'OPP_FT%', 'AST/GM', 'OPP_AST/GM', 'AST/TO', 'OPP_AST/TO',
                          'S+B/GM', 'OPP_S+B/GM',
                          'ARENA_CAP', 'KP_RANK', 'SEED', 'WIN', 'LOSS',
                          'ADJ_EM', 'ADJ_O', 'ADJ_D', 'ADJ_T', 'LUCK',
                          'SOS_ADJ_EM', 'SOS_OPP_O', 'SOS_OPP_D', 'NCSOS_ADJ_EM']]

print(float_rollup.info())

#%%
# DEFINE NUMERIC D-TYPES
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

# IDENTIFY NUMERIC COLUMNS
regular_numeric_cols = regular.select_dtypes(include=numerics)
tourney_numeric_cols = tourney.select_dtypes(include=numerics)
rollup_numeric_cols = rollup_df.select_dtypes(include=numerics)

#%% [markdown]
# Applying normality tests to all three DataFrames:
    # `tourney` `regular` `rollup`

#%%
## NORMALITY TESTS

# K-SQUARED TEST
# Kolmogorov-Smirnov (K-S)
ks_index = ['K-S', 'P-VALUE']
ks_test_regular = pd.DataFrame(index=ks_index)
ks_test_tourney = pd.DataFrame(index=ks_index)
ks_test_rollup = pd.DataFrame(index=ks_index)

# D'AGOSTINO K-SQUARED TEST
da_index = ['DAGOSTINO', 'P-VALUE']
da_test_regular = pd.DataFrame(index=da_index)
da_test_tourney = pd.DataFrame(index=da_index)
da_test_rollup = pd.DataFrame(index=da_index)

# SHAPIRO-WILK TEST
shap_index = ['SHAPIRO-WILK', 'P-VALUE']
shap_test_regular = pd.DataFrame(index=shap_index)
shap_test_tourney = pd.DataFrame(index=shap_index)
shap_test_rollup = pd.DataFrame(index=shap_index)

for col in tourney_numeric_cols.columns:
    ks_test_tourney[col + '_KS']  = st.kstest(tourney_numeric_cols[col], 'norm')
    da_test_tourney[col + '_DA'] = st.normaltest(tourney_numeric_cols[col])
    shap_test_tourney[col + '_Shap'] = st.shapiro(tourney_numeric_cols[col])

for col in regular_numeric_cols.columns:
    ks_test_regular[col + '_KS'] = st.kstest(regular_numeric_cols[col], 'norm')
    da_test_regular[col + '_DA'] = st.normaltest(regular_numeric_cols[col])
    shap_test_regular[col + '_Shap'] = st.shapiro(regular_numeric_cols[col])

for col in float_rollup.columns: #rollup_numeric_cols
    ks_test_rollup[col + '_KS']  = st.kstest(float_rollup[col], 'norm')
    da_test_rollup[col + '_DA'] = st.normaltest(float_rollup[col])
    shap_test_rollup[col + '_Shap'] = st.shapiro(float_rollup[col])

#%%
print(f'K-S NORMALITY TEST:')
print(ks_test_rollup.T) #print(ks_test_tourney.T)
print('*'*75)
print(f'D`AGOSTINO-PEARSON NORMALITY TEST:')
print(da_test_tourney.T) #print(da_test_rollup.T)
print('*'*75)
print(f'SHAPIRO-WILK NORMALITY TEST:')
print(shap_test_tourney.T) #print(shap_test_rollup.T)



#%% [markdown]
# Only a handful of columns showing signs of non-normal distribution:
    # * `Month`, `NumOT`, `Seed_2022`, `Seed`, `KP_Rank`, `KP_Rank_2022`
    # * CONSIDERATION: `LUCK`, `ADJ_EM`, `SOS_ADJ_EM`, `NCSOS_ADJ_EM_KS`
# Fortunately, all of the identified non-normal distributions are explainable
    # * For example, overtime `numOT` is a generally rare real-life scenario
    # * `Month` is skewed as NCAAB schedule is played from October-April each year
    # * All can be safely dropped from dataset without fear of losing valuable insight
# Assessing normality tests for strictly 2022 data within `rollup`:
    # * Aside from `Seed`, all other columns appear to be normally distributed
    # * Consistently low p-values confirm such findings
    # * Certain features with more significant p-values than others

#%%
## ASSIGN NUMERIC X
X_regular = regular[regular._get_numeric_data().columns.to_list()[:-1]]
X_tourney = tourney[tourney._get_numeric_data().columns.to_list()[:-1]]
X_rollup = rollup_df[rollup_df._get_numeric_data().columns.to_list()[:-1]]

print(X_rollup.columns)

#%%
## ASSIGN TARGET VARIABLE
Y_regular = regular['WMargin']
Y_tourney = tourney['WMargin']
Y_rollup = rollup_df['WIN%']

## DROP ID / NON-NORMAL / IRRELEVANT COLUMNS
#X_regular.drop(columns=['NumOT', 'Seed_2022', 'KP_Rank_2022'], inplace=True, axis=1) #'WMargin', 'Seed', 'KP_Rank',
#X_tourney.drop(columns=['NumOT', 'Seed_2022', 'KP_Rank_2022'], inplace=True, axis=1) #'WMargin', 'Seed', 'KP_Rank',
#X_rollup.drop(columns=['Seed', 'KP_Rank'], inplace=True, axis=1)

#%% [markdown]
# Implementing StandardScaler to normalize wide range of metrics found within the data at hand

#%%
## STANDARD SCALER
X_regular = StandardScaler().fit_transform(X_regular)
X_tourney = StandardScaler().fit_transform(X_tourney)
X_rollup = StandardScaler().fit_transform(X_rollup)

print(f'DATA SCALED')

#%%
## PRINCIPAL COMPONENT ANALYSIS
pca = PCA(n_components=20, svd_solver='full') # 'mle'

pca.fit(X_regular)
X_PCA = pca.transform(X_regular)

print('PRINCIPAL COMPONENT ANALYSIS')
print('*'*100)
print('ORIGINAL DIMENSIONS:', X_regular.shape)
print('TRANSFORMED DIMENSIONS:', X_PCA.shape)
print('*'*100)
print(f'EXPLAINED VARIANCE RATIO: {pca.explained_variance_ratio_}')

#%%
x = np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1, 1)

plt.figure(figsize=(12,8))
plt.plot(x, np.cumsum(pca.explained_variance_ratio_))
plt.title('PCA - EXPLAINED VARIANCE RATIO', fontsize=20)
plt.xlabel('COMPONENT #', fontsize=16)
plt.ylabel('EXPLAINED VARIANCE', fontsize=16)
plt.legend(loc='best')
plt.xticks(x)
plt.grid()
plt.show()

#%% [markdown]
## OBSERVATIONS:

# 7 COMPONENTS: 0.55798407 = 0.12247086 + 0.09800916 + 0.0827148 + 0.06972242 + 0.0639338  + 0.06312243 + 0.0580106
    # * 7 unique components account for at least 5.0% of total explained variance
# 15 COMPONENTS: 0.80870685 = 0.55798407 + 0.04888373 + 0.03717191 + 0.03138727 + 0.0302505 + 0.0301666 + 0.02511048 + 0.02412783 + 0.02362446
    # * 15 unique components account for at least 2.0% of total explained variance
# 20 COMPONENTS: 0.90537805 = 0.80870685 + 0.02270043 + 0.02098876 + 0.01847206 + 0.01803448 + 0.01647547
    # * 20 unique components account for 90.54% of total explained variance

#%%
# SINGULAR VALUE DECOMPOSITION ANALYSIS [SVD]
# CONDITIONAL NUMBER

# ORIGINAL DATA
from numpy import linalg as LA

H = np.matmul(X_tourney.T, X_tourney)
_, d, _ = np.linalg.svd(H)
print(f'ORIGINAL DATA: SINGULAR VALUES {d}')
print('*'*100)
print(f'ORIGINAL DATA: CONDITIONAL NUMBER {LA.cond(X_tourney)}')

#%%
# TRANSFORMED DATA
H_PCA = np.matmul(X_PCA.T, X_PCA)
_, d_PCA, _ = np.linalg.svd(H_PCA)
print(f'TRANSFORMED DATA: SINGULAR VALUES {d_PCA}')
print('*'*100)
print(f'TRANSFORMED DATA: CONDITIONAL NUMBER {LA.cond(X_PCA)}')

#%%
# CONSTRUCTION OF REDUCED DIMENSION DATASET
#pca_df = pca.explained_variance_ratio_

a, b = X_PCA.shape
column = []
df_pca = pd.DataFrame(X_PCA).corr()

for i in range(b):
    column.append(f'PCA {i+1}')
sns.heatmap(df_pca,
            annot=True,
            cmap='mako',
            #annot_kws=10,
            xticklabels=column,
            yticklabels=column,
            fmt='.1g',
            linecolor='white',
            linewidth=.1,
            )
plt.title("CORRELATION COEFFICIENT")
plt.xticks(fontsize=10)
plt.show()

#%%
## RESET INDEX
regular.reset_index(inplace=True)
tourney.reset_index(inplace=True)
print(regular.index)
print(tourney.index)

#%%
# 1) Line-plot #1
plt.figure(figsize=(8,6))
sns.lineplot(data=regular,
             x=regular['Season'],
             y=regular['WFGM3'],
             hue=regular['Conference'],
             palette='mako',
             markers=True,
             legend=True,
             )
plt.title('WINNER 3PT ATTEMPTS BY SEASON [2004-2022]', fontsize=20)
plt.xlabel('SEASON', fontsize=16)
plt.ylabel('WINNER 3PT ATTEMPTS', fontsize=16)
plt.xticks(list(range(2003,2023,3)))
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 1) Line-plot #2
plt.figure(figsize=(8,6))
sns.lineplot(data=regular,
             x=regular['Season'],
             y=regular['LFGM3'],
             hue=regular['Conference'],
             palette='mako',
             markers=True,
             legend=True,
             )
plt.title('LOSER 3PT ATTEMPTS BY SEASON [2004-2022]', fontsize=20)
plt.xlabel('SEASON', fontsize=16)
plt.ylabel('LOSER 3PT ATTEMPTS', fontsize=16)
plt.xticks(list(range(2003,2023,3)))
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 2) Bar-plot (stacked, grouped)
plt.figure(figsize=(8,6))

sns.barplot(data=regular_mm, x='Seed', y='Win',
            palette='mako',
            #stack #group #hue='Conference',
            )
plt.title('AVG. WINS/YR BY SEED [2004-2022]', fontsize=20)
plt.xlabel('TOURNAMENT SEED', fontsize=16)
plt.ylabel('AVG. WINS/YR', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 3) Count-plot
plt.figure(figsize=(16,8))
sns.countplot(data=regular, x='Conference',
              palette='mako',
              order=regular['Conference'].value_counts(ascending=False).index)
plt.title('GAMES PLAYED BY CONFERENCE [2004-2022]', fontsize=20)
plt.xlabel('CONFERENCE', fontsize=16)
plt.ylabel('TOTAL GAMES PLAYED', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 4) Cat-plot

cat_vars_regular_W = regular[[#'Win', #'Seed'
                            'WScore', 'WOR', 'WDR',  #'WScore',
                            'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                            ]]

cat_vars_regular_L = regular[[#'Lose', #'Seed',
                            'LScore', 'LOR', 'LDR', #'LScore',
                            'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
                            ]]

cat_vars_tourney_W = tourney[[#'Win', #'Seed',
                            'WScore', 'WOR', 'WDR', #'WScore',
                            'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',
                            ]]

cat_vars_tourney_L = tourney[['LScore', 'LOR', 'LDR', #'LScore',
                              'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',]] #'Lose', #'Seed',


#plt.subplots(1,2)
plt.figure(figsize=(16,12))
sns.catplot(data=cat_vars_regular_W, #data=regular, #WFG%
            palette='mako',
            kind='strip',
            #legend=True,
            )
plt.title('WINNING TEAM STATISTICS', fontsize=16)
plt.xlabel('STATISTIC (WINNING TEAM)', fontsize=12)
plt.ylabel('FREQUENCY', fontsize=12)
plt.legend(loc='best')
plt.grid()
plt.tight_layout(pad=1)

plt.show();

#%%
sns.catplot(data=cat_vars_regular_L, #data=regular, #WFG%
            palette='mako',
            kind='strip',
            #legend=True,
            )
plt.title('LOSING TEAM STATISTICS', fontsize=16)
plt.xlabel('STATISTIC (LOSING TEAM)', fontsize=12)
plt.ylabel('FREQUENCY', fontsize=12)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 5) Pie-chart
phylum = regular['tax_phylum']

tax_phylums = ['Chordata', 'Pinophyta', 'Arthropoda',
              'Tracheophyta', 'Magnoliophyta', 'Tracheophyta',
              'Reptilia', 'Euarthropoda', 'None']

labels = tax_phylums
explode = [.03, .03, .03, .03, .03, .03, .03, .03, .03]

fig, ax = plt.subplots(1,1)

ax.pie(x=regular['tax_phylum'].value_counts(),
       labels=labels,
       explode=explode,
       rotatelabels=True,
       normalize=True,
       autopct='%1.1f%%',) # pctdistance=.6
#ax.axis('square')
plt.title('TEAM MASCOTS BY TAX. PHYLUM', fontsize=20)
#plt.xlabel('TBU', fontsize=16)
#plt.ylabel('TBU', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=.25)

plt.show()


#%%
# 6) Dis-plot
plt.figure(figsize=(12,6))
sns.displot(data=tourney['WMargin'], #, x='Seed',
            palette='mako',
            kind='hist',
            element='step',
            discrete=False,
            legend=True,
            )
plt.title('WIN MARGIN DISTRIBUTION [2004-2022]', fontsize=14)
plt.xlabel('WIN MARGIN', fontsize=12)
plt.ylabel('FREQUENCY', fontsize=12)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 7) Pair plot
pair_vars = tourney[['WScore', 'WOR', 'WDR',
'WAst', 'WTO', 'WStl', 'WBlk', 'WPF',]]

plt.figure(figsize=(12,12))
sns.pairplot(pair_vars, palette='mako')
#plt.title('PAIRPLOT [2004-2022]', fontsize=20)
#plt.xlabel('TBU', fontsize=16)
#plt.ylabel('TBU', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 8) Heatmap
# create correlation variables relative to rest of DataFrame
margin_corr = tourney.corr()[['WMargin']].sort_values(by='WMargin', ascending=False)

# create heatmap to visualize correlation variable
plt.figure(figsize=(8, 8))
sns.heatmap(margin_corr, annot=True, cmap='mako', vmin=-1, vmax=1, linecolor='white', linewidth=2)

plt.title('WIN MARGIN CORRELATION [2004-2022]', fontsize=20)
plt.xlabel('WIN MARGIN CORRELATION', fontsize=16)
plt.ylabel('FEATURES', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()


#%%
# 9) Hist-plot
plt.figure(figsize=(16,8))
sns.histplot(x=rollup_df['STATE'],
             palette='mako',
             hue=rollup_df['CONF'],
             bins=30,
             binwidth=3,
             legend=True,
             multiple='stack',
             )
plt.title('TEAM HOME STATE DISTRIBUTION (BY CONFERENCE)', fontsize=20)
plt.xlabel('TEAM HOME STATE', fontsize=16)
plt.ylabel('FREQUENCY', fontsize=16)
plt.legend(loc='best')

plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 10) QQ-plot
plt.figure(figsize=(8,6))
qqplot(tourney['Venue_Capacity'])
plt.title('HOME VENUE ATTENDANCE CAPACITY', fontsize=16)
plt.xlabel('HOME VENUE CAPACITY', fontsize=12)
plt.ylabel('FREQUENCY', fontsize=12)
plt.show()

#%%
# 11) Kernel density estimate
kde_vars_tourney = tourney[['WMargin', 'WScore', 'LScore',]]

sns.kdeplot(data=kde_vars_tourney,
            bw_adjust=.2,
            cut=0,
            #hue='Month',
            multiple='stack', #fill
            )

plt.title('W/L SCORES vs. MARGIN (KDE)', fontsize=16)
plt.xlabel('POINTS SCORED / MARGIN', fontsize=12)
plt.ylabel('DENSITY', fontsize=12)
plt.grid()
plt.tight_layout(pad=1)

plt.show()

#%%
# 13) Multivariate Box plot
box_vars_tourney = regular[['WTO', 'WStl', 'WBlk',
                            'Conference', 'WScore', 'WAst',]]

plt.figure()
sns.boxplot(x=box_vars_tourney['WAst'],
            y=box_vars_tourney['WTO'],
            #hue=box_vars_tourney['Month'],
            palette='mako',
            )

plt.title('ASSISTS vs. TURNOVERS', fontsize=16)
plt.xlabel('TEAM ASSISTS / GM', fontsize=12)
plt.ylabel('TEAM TURNOVERS / GM', fontsize=12)
plt.grid()
plt.tight_layout(pad=1)

plt.show()

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

#%% [markdown]
## DATA DICTIONARY:

# NCAA CONFERENCES
    # 'A10', 'AAC' 'ACC', 'AE', 'AS'
    # 'BIG10', 'BIG12', 'BIGEAST', 'BIGSKY', 'BIGSOUTH', 'BIGWEST'
    # 'COLONIAL', 'CUSA', 'HORIZON', 'IVY'
    # 'MAAC', 'MAC', 'MEAC', 'MVC', 'MWC', 'NE'
    # 'OVC', 'PAC12', 'PATRIOT', 'SEC', 'SOUTHERN', 'SOUTHLAND',
    # 'SUMMIT', 'SUNBELT', 'SWAC', 'WAC', 'WCC'

# DATETIME FEATURES OF INTEREST
    # 'Date', 'Month'

# NUMERIC FEATURES OF INTEREST
    # 'Win', 'Loss', 'Seed'. 'WScore', 'LScore', 'WMargin',
    # 'Venue_Capacity', 'WFG%', 'LFG%', 'NumOT'
    # 'KP_Rank', 'KP_ADJ_EM', 'KP_ADJ_O', 'KP_ADJ_D',

# CATEGORICAL FEATURES OF INTEREST
    # 'Conference', 'Mascot', 'City', 'State', 'Venue',
    # 'tax_family', 'tax_order', 'tax_class', 'tax_phylum',
    # 'tax_kingdom', 'tax_domain',

# COLUMNS
    # * ['Season', 'DayNum', 'Date', 'Month', 'WTeamID', 'Conference', 'Win',
#        'Loss', 'Seed_2022', 'KP_Rank_2022', 'Seed', 'KP_Rank', 'KP_ADJ_EM',
#        'KP_ADJ_O', 'KP_ADJ_D', 'Mascot', 'Mascot2', 'City', 'State', 'Venue',
#        'Venue_Capacity', 'tax_family', 'tax_order', 'tax_class', 'tax_phylum',
#        'tax_kingdom', 'tax_domain', 'WScore', 'LTeamID', 'LScore', 'WLoc',
#        'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR',
#        'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3',
#        'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF',
#        'WFG%', 'LFG%', 'WMargin']'

#%%