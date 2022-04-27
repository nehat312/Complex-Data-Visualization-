#%% [markdown]
# DATS-6401 - LAB #6
# Nate Ehat

#%% [markdown]
# In this Lab, you will learn how to convert non-gaussian distributed dataset into a gaussian distributed dataset.

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
import scipy.stats as st

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.gofplots import qqplot

print("\nIMPORT SUCCESS")

#%%
# 1. Generate a random data (x) with 5000 samples and normal distribution (mean = 0, std = 1).
# Then use np.cumsum to convert the generated normal data into a non-normal distributed data (y).
# Graph the normal (x) and non-normal (y) data set versus number of samples and histogram plot of the normal(x) and non-normal (y) dataset
# 2x2 figure using subplot.
# Number of bins = 100. Figure size = 9,7.
# Add grid and appropriate x-label, y-label, and title to the graph.

x = np.random.randn(5000)
y = np.cumsum(x)

fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000), y=x, ax=axes[0,0]).set(xlabel= '# of samples', ylabel='Magnitude')
axes[0,0].set_title('Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000), y=y, ax=axes[0,1]).set(xlabel= '# of samples', ylabel='Magnitude')
axes[0,1].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.histplot(x=x, bins=100, ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=y, bins=100, ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram Non-Gaussian data')
plt.tight_layout()
plt.show()


#%%
# 2. Perform a K-S Normality test on the x and y dataset [dataset generated in the previous question].
# Display the p-value and statistics of the test for the x and y [a separate test is needed for x and y].
# Interpret the K-S test [Normal or Not Normal with 99% accuracy] by looking at the p-value.
# Display the following information on the console:
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  x dataset looks ______
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  y dataset looks ______

kstest_x = st.kstest(x,'norm')
kstest_y = st.kstest(y,'norm')

print(f"K-S test: statistics={kstest_x[0]:.5f}, p-value={kstest_x[1]:.5f}")
print(f"K-S test: x dataset looks {'Normal' if kstest_x[1] > 0.01 else 'Non-Normal'}")
print(f"K-S test: statistics={kstest_y[0]:.5f}, p-value={kstest_y[1]:.5f}")
print(f"K-S test: y dataset looks {'Normal' if kstest_y[1] > 0.01 else 'Non-Normal'}")


#%%

# 3. Repeat Question 2 with the “Shapiro test”.
    # Shapiro test: statistics= _____ p-value = ______
    # Shapiro test: x dataset looks ______
    # Shapiro test: statistics= _____ p-value = ______
    # Shapiro test: y dataset looks ______

shapiro_test_x = st.shapiro(x)
shapiro_test_y = st.shapiro(y)

print(f"Shapiro test: statistics={shapiro_test_x[0]:.5f}, p-value={shapiro_test_x[1]:.5f}")
print(f"Shapiro test: x dataset looks {'Normal' if shapiro_test_x[1] > 0.01 else 'Non-Normal'}")
print(f"Shapiro test: statistics={shapiro_test_y[0]:.5f}, p-value={shapiro_test_y[1]:.5f}")
print(f"Shapiro test: y dataset looks {'Normal' if shapiro_test_y[1] > 0.01 else 'Non-Normal'}")


#%%

# 4. Repeat Question 2 with the “D'Agostino's 𝐾2 test”.
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______

da_test_x = st.normaltest(x)
da_test_y = st.normaltest(y)

print(f"da_k_squared test: statistics={da_test_x[0]:.5f}, p-value={da_test_x[1]:.5f}")
print(f"da_k_squared test: x dataset looks {'Normal' if da_test_x[1] > 0.01 else 'Non-Normal'}")
print(f"da_k_squared test: statistics={da_test_y[0]:.5f}, p-value={da_test_y[1]:.5f}")
print(f"da_k_squared test: y dataset looks {'Normal' if da_test_y[1] > 0.01 else 'Non-Normal'}")


#%%
# 5. Convert the non-normal data (y) to normal using the rankdata and norm.ppf.
# Add appropriate x- label, y-label, title, and grid to the 2x2 subplot graph.
# The final graph should look like bellow.

new_y = st.norm.ppf(st.rankdata(y)/(len(y) + 1))

fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000), y=y, ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000), y=new_y, ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=y, bins=100, ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_y, bins=100, ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

#%%
# 6. Plot the QQ plot of the y and the y transformed.
# The final plot should be like below.

fig, ax = plt.subplots(ncols=2)
qqplot(y, ax=ax[0])
ax[0].set_title('y Data: Non-Normal')
qqplot(new_y, ax=ax[1])
ax[1].set_title('Transformed y: Normal')
plt.show()


#%%
# 7. Perform a K-S Normality test on the y transformed dataset.
# Display the p-value and statistics of the test for y transformed.
# Interpret the K-S test [Normal or Not Normal with 99% accuracy] by looking at p-value.
# Display the following information on the console:
    # K-S test: statistics= _____ p-value = ______
    # K-S test:  y transformed dataset looks ______
kstest_new_y = st.kstest(new_y,'norm')

print(f"K-S test: statistics={kstest_new_y[0]:.5f}, p-value={kstest_new_y[1]:.5f}")
print(f"K-S test: new_y dataset looks {'Normal' if kstest_new_y[1] > 0.01 else 'Non-Normal'}")


#%%
# 8. Repeat Question 7 with the “Shapiro test”.
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______

shapiro_test_new_y = st.shapiro(new_y)

print(f"Shapiro test: statistics={shapiro_test_new_y[0]:.5f}, p-value={shapiro_test_new_y[1]:.5f}")
print(f"Shapiro test: new_y dataset looks {'Normal' if shapiro_test_new_y[1] > 0.01 else 'Non-Normal'}")

#%%
# 9. Repeat Question 7 with the “D'Agostino's 𝐾2 test"
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: x dataset looks ______
    # da_k_squared test: statistics= _____ p-value = ______
    # da_k_squared test: y dataset looks ______

da_test_new_y = st.normaltest(new_y)

print(f"da_k_squared test: statistics={da_test_new_y[0]:.5f}, p-value={da_test_new_y[1]:.5f}")
print(f"da_k_squared test: new_y dataset looks {'Normal' if da_test_new_y[1] > 0.01 else 'Non-Normal'}")

#%%
# 10. Do all 3-normality tests confirm the Normality of the transformed data?
# Explain your answer if there is a discrepancy.

#%% [markdown]

# After implementing the normalization process:
# It appears all three normality tests confirm the data is now normalized.

#%%