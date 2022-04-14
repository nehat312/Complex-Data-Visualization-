#%% [markdown]
# DATS-6401 - CLASS 4/13/22
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import pandas_datareader as web


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from scipy import stats as st
import statistics
# import datetime as dt
# from statsmodels.graphics.gofplots import qqplot

print("\nIMPORT SUCCESS")

#%%

## IN-CLASS WHITEBOARD Z-Score CALCS

# 500 americans
mean = 194
std = 11.2

low = 175
high = 225

print(f'Probability that observations fall between '
      f'{low} and {high} is:'
      f' {st.norm(mean, std).cdf(high)-st.norm(mean,std).cdf(low):.2f}')

# 175-225

# 227.2 -
# 216.4 -
# 205.2 -
# 194 -
# 182.8 -
# 171.6 -
#



#%%
#%%

## IN-CLASS WHITEBOARD Z-Score CALCS

# 500 americans
mean = 194
std = 11.2

low = 175
high = 225

print(f'Probability that observations fall between '
      f'{low} and {high} is:'
      f' {st.norm(mean, std).cdf(high)-st.norm(mean,std).cdf(low):.2f}')

#%%



#%%



#%%