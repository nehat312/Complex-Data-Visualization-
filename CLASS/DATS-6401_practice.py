#%% [markdown]
#### WILLARD REAL ESTATE
    # * 

#%%
# LIBRARY IMPORTS
import os
import numpy as np
import pandas as pd
import mysql.connector
from mysql.connector import Error

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics
import datetime as dt

from statsmodels.formula.api import ols
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from statsmodels.formula.api import glm
# import statsmodels.formula.api as smf  # Support for formulas

from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV #LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import LinearSVC #SVC

#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.metrics import accuracy_score
#from sklearn.metrics import confusion_matrix 
#from sklearn.metrics import classification_report

#np.set_printoptions(threshold=200)

print("\nIMPORT SUCCESS")


#%%

list1 = list(range(0,1001,1))
np.random(list1)

#%%

n = 1000
meanx = 0
meany = 2

stdx = np.sqrt(1)
stdy = np.sqrt(5)

x = np.random.normal(meanx, stdx, n)
y = np.random.normal(meany, stdy, n)

#%%

plt.figure(figsize=(12,12))
plt.plot(x, 'b', label="X")
plt.plot(x, 'o', label="Y")
plt.legend()
plt.grid()
plt.show()

#%%
From toolbox import correlation_coefficient_
r_x_y = correlation_coefficient_ ()

