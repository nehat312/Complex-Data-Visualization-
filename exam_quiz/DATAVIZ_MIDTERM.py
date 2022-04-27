CLASS NOTES
PLOTS
Stacked bar charts -  at least one categorical variable
IQR Method
Inter-Quartile Range
Box-plot / outlier detection
‘Whisker plot’
QQPlot
Test of normality
Allows for parametric methods
Ideal = 45 degree lines
Outliers are anything outside +/- 1.5 quartiles away from Median
Medians serve as a measure of central tendency
Covariance
Sigma X = std dev X
Used to determine Pearson correlation coefficient
MATPLOTLIB
Comparable to MatLab
Z-Transformation / Scaling
Z = (y - y preds) / std(y)
df2_z_score = (df2-df2.mean()) / df2.std()
Axis Selection
X = Independent Variable
Y = Dependent Variable
Impute Missing Values
loc()
iloc() - better for categorical
Gaussian vs Non-Gaussian
Essentially indicates normal distribution

CLASS CODE 2/9/22

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.graphics.gofplots import qqplot

import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%


fig, ax = plt.subplots(1,1)

labels = ['C', 'C++', 'Java', 'Python', 'PHP']
score_men = [23, 17, 35, 29, 12]
score_women = [35, 25, 18, 15, 38]
explode = [.03, .03, .3, .03, .03]

#%%
ax.pie(score_men, labels=labels, explode=explode, autopct='%1.2f%%')
ax.axis('square')
plt.show()

#%%
plt.figure(figsize=(8,8))
plt.bar(labels, score_men, label='Men')
plt.bar(labels, score_women, label='Women', bottom=score_men)
plt.title('SAMPLE SCORES')
plt.xlabel('Language')
plt.ylabel('Score')
plt.legend(loc='best')
plt.show()

#%%
width = 0.4
x = np.arange(len(labels))
ax.bar(x - width/2, score_men, width, label='Men')
ax.bar(x + width/2, score_women, width, label='Women')
ax.set_xlabel('Language')
ax.set_ylabel('Score')
ax.set_yticks(x)
ax.set_yticklabels(labels)
ax.set_title('MEN VS WOMEN')
ax.legend()
plt.show()


#%%
np.random.seed(10)
data = np.random.normal(100, 20, 1000)
plt.figure()
plt.boxplot(data)
plt.xticks([1])
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('BOXPLOT')
plt.show()

#%%
plt.figure()
plt.hist(data, bins=50)
plt.show()


#%%
np.random.seed(10)
data1 = np.random.normal(100, 10, 1000)
data2 = np.random.normal(90, 20, 1000)
data3 = np.random.normal(80, 30, 1000)
data4 = np.random.normal(70, 40, 1000)
data = [data1, data2, data3, data4]

plt.figure()
plt.boxplot(data)
plt.xticks([1,2,3,4])
plt.ylabel('Average')
plt.xlabel('Data Number')
plt.grid()
plt.title('BOXPLOT')
plt.show()


#%%
plt.figure()
plt.hist(data, bins=50)
plt.show()

#%%
data1 = np.random.normal(0, 1, 1000)
plt.figure()
qqplot(data1, line='45')
plt.title('QQ-PLOT')
plt.show()

#%%
plt.figure(figsize=(12,8))
x = range(1,6)
y = [1,4,6,8,4]
plt.plot(x, y, color = 'blue', lw=3)
plt.fill_between(x, y, label='Area', alpha=.3)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Simple Area Plot')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%

x = np.linspace(0, 2*np.pi, 41)
y = np.exp(np.sin(x))

(markers, stemlines, baseline) = plt.stem(y,
                                          )
plt.stem(markers, label='exp(sin(x))', color='red')
plt.setp(baseline, color='gray, lw=2, linestyle='-')




CLASS CODE 2/2/22

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as web

# import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%
plt.style.use('fivethirtyeight')
    # ggplot, bmh, fivethirtyeight, seaborn-darkgrid, seaborn-whitegrid, seaborn-deep
x = np.linspace(0, 2*np.pi, 20)
y1 = np.sin(x)
y2 = np.cos(x)

font1 = {'family':'serif', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'darkred', 'size':30}

plt.figure(figsize=(12,8))
plt.plot(x,y1, lw=4, label='sin(x)', color='r', marker='o', ms=20, mec='k', mfc='b')
plt.plot(x,y2, lw=4, label='cos(x)', color='c', marker='*', ms=20, mec='k', mfc='r')
plt.legend(loc='best', fontsize=20)
plt.title('sin(x) vs cos(x)', fontdict=font2, loc='left')
plt.grid(axis='x')
plt.xlabel('Samples', fontdict=font1)
plt.ylabel('Magnitude', fontdict=font1)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.show()


#%%
# VARIABLE ASSIGNMENT
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2000-01-01'
end_date = '2022-02-02'
#n = 0

print("\nVARIABLES ASSIGNED")

#%%
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)
msft_cols = msft.columns
print(msft_cols)

#%%
plt.figure(figsize=(16,8))
plt.subplot(2, 3, 1)
plt.plot(msft.Close)
plt.grid(axis='x')
plt.title('MSFT CLOSE', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

plt.subplot(2, 3, 2)
plt.plot(msft.High)
plt.grid(axis='x')
plt.title('MSFT HIGH', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

plt.subplot(2, 3, 3)
plt.plot(msft.Low)
plt.grid(axis='x')
plt.title('MSFT LOW', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

plt.subplot(2, 3, 4)
plt.plot(msft.Volume)
plt.grid(axis='x')
plt.title('MSFT VOLUME', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('Share Volume', fontdict=font1)

plt.subplot(2, 3, 5)
plt.plot(msft['Adj Close'])
plt.grid(axis='x')
plt.title('MSFT ADJ CLOSE', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

plt.subplot(2, 3, 6)
plt.plot(msft.Open)
plt.grid(axis='x')
plt.title('MSFT OPEN', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

plt.tight_layout(pad=1)
plt.show()

#%%
# SUBPLOTS - ALTERNATE METHOD
fig = plt.figure(figsize=(16,8))

for i in range(1,7):
    ax1 = fig.add_subplots(2,3,i)
    plt.plot(msft[msft_cols[i]])
    ax1.set_xlabel()

if msft_cols[i-1] = 'Volume':
    TBU
else:
    ax1.set_xlabel('USD ($)', fontsize = 15)

#%%

fig, axs = plt.subplots(2, 3, figsize=(12,9))
#z=0
for i in range(1,3):
    for j in range(1,4):
        axs[i-1, j-1].plot(msft[msft_cols][z].values)
        axs[i - 1, j - 1].legend(loc='best')
        axs[i-1, j-1].set_title(f'MSFT {msft_cols[i+j-2]}', fontsize=20)
        axs[i-1, j-1].set_xlabel('Date / Time', fontsize=20)
        axs[i-1, j-1].set_ylabel('$ / Share', fontsize=20)
        axs[i - 1, j - 1].grid(axis='x')
        z=+1

    plt.show()

#%%
axs[0,1] = plt.plot(msft.High)
plt.grid(axis='x')
plt.title('MSFT HIGH', fontdict=font2)
plt.xlabel('Date / Time', fontdict=font1)
plt.ylabel('$ / Share', fontdict=font1)

axs = plt.plot(msft.Low)
plt.grid(axis='x')
axs.title('MSFT LOW', fontdict=font2)
axs.set_xlabel('Date / Time', fontdict=font1)
axs.set_ylabel('$ / Share', fontdict=font1)

axs = plt.plot(msft.Volume)
plt.grid(axis='x')
axs.title('MSFT VOLUME', fontdict=font2)
axs.set_xlabel('Date / Time', fontdict=font1)
axs.set_ylabel('Share Volume', fontdict=font1)

axs = plt.plot(msft.Open)
plt.grid(axis='x')
axs.title('MSFT OPEN', fontdict=font2)
axs.set_xlabel('Date / Time', fontdict=font1)
axs.set_ylabel('$ / Share', fontdict=font1)

axs = plt.plot(msft['Adj Close'])
plt.grid(axis='x')
plt.title('MSFT ADJ CLOSE', fontdict=font2)
axs.set_xlabel('Date / Time', fontdict=font1)
axs.set_ylabel('$ / Share', fontdict=font1)
axs.xticks(range(2000,2021,1))

plt.tight_layout()
plt.show()


#%%

dol_cols1 = msft_cols.drop('Volume')
print(dol_cols1)

#%%

dol_cols = ['High', 'Low', 'Open', 'Close', 'Adj Close']
plt.title('MSFT ADJ CLOSE', fontdict=font2)

plt.plot(msft[dol_cols])

plt.xlabel('Date / Time')
plt.ylabel('$ / Share')
plt.legend()
plt.show()


#%%

plt.hexbin(msft.Volume.values, msft.Close.values, gridsize=(50,50))
plt.show()

#%%
correlation = msft.corr()
print(correlation)

#%%

plt.style.use('fivethirtyeight')
np.random.seed(123)
x = np.random.normal(size=5000)
y = 2*x + np.random.normal(size=5000)

plt.figure()
plt.hexbin(x,y,gridsize=(100,100))
plt.xlabel('Random Variable X')
plt.show()

plt.savefig('Test.png', dpi=600)

#%%
pd.plotting.scatter_matrix(msft, diagonal='hist', hist_kwds={'bins':50})
plt.show()






CLASS CODE 1/26/22


#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%
np.random.seed(123)
data_1D = pd.Series([1,2,3,4,np.nan])
index = np.arange(1,7)
data_2D = pd.DataFrame(data=np.random.randn(6,4),
                       index=index, columns=list('ABCD'))
print(data_1D)
print(data_2D)

#%%
data_2D.info()

#%%
df_arrest = pd.DataFrame({'Gender':['female', 'female', 'male', 'male', 'male'],
                   'Age':['25', '18', '', '52', '33'],
                   'Weight':['250', '180', np.nan, '210', '330'],
                   'Location':['CA', 'DC', 'VA', 'MA', 'VA'],
                   'Arrest Record':['No', 'Yes', 'Yes', 'No', np.nan]
                   })

print(df_arrest.head())

#%%
df = sns.load_dataset('car_crashes')
print(df.head())

#name = sns.get_dataset_names()
#print(name)

#%%
# z-score

df2 = df[['speeding', 'alcohol', 'ins_premium', 'total']]
df2_z_score = (df2-df2.mean()) / df2.std()

plt.figure()
df2_z_score[30:].plot()
plt.show()


#%%
col_names = df.columns
total = df(col_names)[0].values

plt.figure(figsize=(12,12))
df.plot[:10](y='total')


#%%
total_speeding_corr_coef = np.corrcoef(df['total'], df['speeding'])[0,1]
total_alcohol_corr_coef = np.corrcoef(df['total'], df['alcohol'])[0,1]
total_prem_corr_coef = np.corrcoef(df['total'], df['ins_premium'])[0,1]

print(f'Sample Pearson Correlation Coefficient between Total & Speed: {total_speeding_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Total & Alcohol: {total_alcohol_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Total & Ins. Prem.: {total_prem_corr_coef:.5f}')


#%%
total_crash = df.total.values
ins_prem = df.ins_premium.values
alcohol = df.alcohol.values
speeding = df.speeding.values


#%%

plt.figure(figsize=(12,12))
plt.scatter(total_crash, alcohol)
plt.title(f'CORR COEF')
plt.xlabel('SPEED')
plt.ylabel('Crashes')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
plt.figure(figsize=(12,12))
plt.scatter(total_crash, ins_prem)
plt.title(f'CORR COEF')
plt.xlabel('SPEED')
plt.ylabel('Crashes')
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
# TITANIC DATA SET
titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.info())

#%%
survived_status = titanic.iloc[:,0]
print(survived_status)
#%%
print(titanic.info())

#%%
titanic_males = titanic[titanic['sex'] == 'male']
titanic_females = titanic[titanic['sex'] == 'female']
titanic_females_survived = titanic[(titanic['survived'] == 1) & (titanic['sex'] == 'female')]

titanic_50_plus = titanic[titanic['age'] >= 50]
titanic_49_under = titanic[titanic['age'] <= 49]

print(titanic_males.info()) # 577 males
print(titanic_females.info()) # 317 males
print(titanic_50_plus.info()) # 74 50 or older
print(titanic_49_under.info()) # 640 49 or under
print(titanic_females_survived.info()) # 640 49 or under

#%%
df.dropna(how='any', inplace=True)
#df.fillna()
#

#%%
titanic['age_status'] = np.where(titanic['age'] < 18, 'teen', 'adult')

print(titanic.head())

#df.rename(columns=titanic['age_status'])

#%%

titanic.drop('parch', axis=1, inplace=True)
print(titanic.columns)

#%%

titanic['age_over_mean'] = titanic['age'] / np.mean(titanic['age'])
titanic.insert(loc=4, column='age_over_mean')
print(titanic.head())

#%%
# DIAMONDS DATA SET
diamonds = sns.load_dataset('diamonds')
print(diamonds.head())
print(diamonds.info())

print(diamonds['clarity'].unique())
print(diamonds['color'].unique())

#%%
print(diamonds.pop('carat'))

#%%
index = np.arange(10, len(df))
df1 = diamonds.drop(index)

#%%





CLASS CODE 1/19/22

#%%
# LIBRARY IMPORTS
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics
import datetime as dt

print("\nIMPORT SUCCESS")

#%%

df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tips.csv')
df.head()

#%%
col_names = df.columns
col_names

#%%
df.describe()

#%%
total_bill_mean = np.mean(df['total_bill'])
tip_mean = np.mean(df['tip'])
total_bill_var = np.var(df['total_bill'])
tip_var = np.var(df['tip'])
total_bill_med = np.median(df['total_bill'])
tip_med = np.median(df['tip'])
total_bill_std = np.std(df['total_bill'])
tip_std = np.std(df['tip'])
total_bill_cov = np.cov(df['total_bill'])
tip_cov = np.cov(df['tip'])

print(f'Total Bill Mean: {total_bill_mean:.2f}')
print(f'Tip Mean: {tip_mean:.2f}')
print(f'Total Bill Variance: {total_bill_var:.2f}')
print(f'Tip Variance: {tip_var:.2f}')
print(f'Total Bill Median: {total_bill_med:.2f}')
print(f'Tip Median: {tip_med:.2f}')
print(f'Total Bill Std. Dev.: {total_bill_std:.2f}')
print(f'Tip Std. Dev.: {tip_std:.2f}')
print(f'Total Bill Cov.: {total_bill_cov:.2f}')
print(f'Tip Cov.: {tip_cov:.2f}')

#%%
# CORRELATION COEFFICIENT CALC
## *** NEED TO FIX THIS ***
## SELF-CALCULATE

n = 50
meanx = tip_mean
meany = total_bill_mean

stdx = tip_std
stdy = total_bill_std

x = np.random.normal(meanx, stdx, n)
y = np.random.normal(meany, stdy, n)

#%%

plt.figure(figsize=(12,12))
plt.hist(x)
plt.hist(y)
plt.title(f'HISTOGRAM PLOT:')# {r_x_y:.2f}')
plt.ylabel('TOTAL BILL')
plt.xlabel('TOTAL TIP')
plt.legend()
plt.grid()
plt.show()

#%%
#from toolbox import correlation_coefficient_calc

#r_x_y = correlation_coefficient_calc(tip, meal)
print(f'Correlation Coefficient - Bill vs Tip: {r_x_y:.2f}')


#%%
cov(X, Y) = (sum (x - np.mean(x)) * (y - np.mean(y)) ) * 1/(n-1)


#%%
#!pip install pandas_datareader

#%%
import pandas_datareader as web

#%%
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
df = web.DataReader('TSLA', data_source='yahoo', start='2000-01-01', end='2022-01-18')
df.describe()

#%%
stock_cols = df.columns
stock_cols

#%%

plt.figure(figsize=(12,12))
plt.plot(df['Adj Close'])
plt.title('TSLA Adjusted Close Price')
plt.xlabel('Date')
plt.ylabel('Year')
#plt.xticks(df.index)
#plt.xticks(f'{range(df['Adj Close']):mm/;

#%%

#%%

#%%

# idxmax


#%%

### SEPARATE SECTION OF CLASS - TEST LAB
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats as stats
import statistics
import datetime as dt

print("\nIMPORT SUCCESS")

#%%
data = np.random.randn(4,5)
print(data)


#%%

df = pd.DataFrame(data=data, columns=[['A', 'B', 'C', 'D', 'E']], index=['Monday', 'Tuesday', 'Wednesday', 'Thursday'])
df.head()

# %%

df3 = df.copy()
for i in range(len(df)):
    df3['max'] = df.astype('float64').idxmax(axis=1)
    df3['min'] = df.astype('float64').idxmin(axis=1)
    df3.loc['max'] = df.astype('float64').idxmax(axis=0)
    df3.loc['min'] = df.astype('float64').idxmin(axis=0)

#MANUALLY REMOVE CORNER EMPTY SUBTOTALS

#%%
df3.head()

#%%


#%%
## SCRATCH - MANUAL WAY
A_max = np.max(df['A'])
B_max = np.max(df['B'])
C_max = np.max(df['C'])
D_max = np.max(df['D'])
E_max = np.max(df['E'])

# %%





LAB #2

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import pandas_datareader as web

#import matplotlib.pyplot as plt
#from tabulate import tabulate

# import seaborn as sns
# from scipy import stats as stats
# import statistics

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1
# Using the pandas_datareader package connect to yahoo database
# Load the stock value for the following giant companies:
    # Stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
# You will need the following package to be able to connect to yahoo API.
# Make sure to use the updated version of the pandas and pandas' data_reader
# Pick the start date as '2000-01-01' and the end date to be Sep 8th , 2021.

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2000-01-01'
end_date = '2021-09-08'

print("\nVARIABLES ASSIGNED")

#%%
# QUESTION 2
# The database contains the stock values of 6 major giant companies.
# Each company dataset contains 6 features:
# "High", "Low", "Open", "Close", "Volume", "Adj Close" in USD($).
# Load the data set and create a table as shown below for the mean of each attribute.
# Display the table of the console.
# There are multiple ways to create a table in python. Pick a method of your choice.

# Pull ticker data
aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)

stock_pulls = [aapl, orcl, tsla, ibm, yelp, msft]

print("\nSTOCKS PULLED")

#%%
# Generate empty DataFrames
df_mean = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_median = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_std = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])
df_var = pd.DataFrame(columns = ['Ticker', 'High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close'])

print("\nDATAFRAMES GENERATED")

#%%
# Calculate mean values for each ticker utilizing NumPy functions
aapl_mean = ["AAPL", round(np.mean(aapl['High']), 2), round(np.mean(aapl['Low']), 2),
              round(np.mean(aapl['Open']), 2), round(np.mean(aapl['Close']), 2),
              round(np.mean(aapl['Volume']), 2), round(np.mean(aapl['Adj Close']), 2)]

orcl_mean = ["ORCL", round(np.mean(orcl['High']), 2), round(np.mean(orcl['Low']), 2),
              round(np.mean(orcl['Open']), 2), round(np.mean(orcl['Close']), 2),
              round(np.mean(orcl['Volume']), 2), round(np.mean(orcl['Adj Close']), 2)]

tsla_mean = ["TSLA", round(np.mean(tsla['High']), 2), round(np.mean(tsla['Low']), 2),
              round(np.mean(tsla['Open']), 2), round(np.mean(tsla['Close']), 2),
              round(np.mean(tsla['Volume']), 2), round(np.mean(tsla['Adj Close']), 2)]

ibm_mean = ["IBM", round(np.mean(ibm['High']), 2), round(np.mean(ibm['Low']), 2),
             round(np.mean(ibm['Open']), 2), round(np.mean(ibm['Close']), 2),
             round(np.mean(ibm['Volume']), 2), round(np.mean(ibm['Adj Close']), 2)]

yelp_mean = ["YELP", round(np.mean(yelp['High']), 2), round(np.mean(yelp['Low']), 2),
              round(np.mean(yelp['Open']), 2), round(np.mean(yelp['Close']), 2),
              round(np.mean(yelp['Volume']), 2), round(np.mean(yelp['Adj Close']), 2)]

msft_mean = ["MSFT", round(np.mean(msft['High']), 2), round(np.mean(msft['Low']), 2),
              round(np.mean(msft['Open']), 2), round(np.mean(msft['Close']), 2),
              round(np.mean(msft['Volume']), 2), round(np.mean(msft['Adj Close']), 2)]

# Calculate median values for each ticker utilizing NumPy functions
aapl_median = ["AAPL", round(np.median(aapl['High']), 2), round(np.median(aapl['Low']), 2),
              round(np.median(aapl['Open']), 2), round(np.median(aapl['Close']), 2),
              round(np.median(aapl['Volume']), 2), round(np.median(aapl['Adj Close']), 2)]

orcl_median = ["ORCL", round(np.median(orcl['High']), 2), round(np.median(orcl['Low']), 2),
              round(np.median(orcl['Open']), 2), round(np.median(orcl['Close']), 2),
              round(np.median(orcl['Volume']), 2), round(np.median(orcl['Adj Close']), 2)]

tsla_median = ["TSLA", round(np.median(tsla['High']), 2), round(np.median(tsla['Low']), 2),
              round(np.median(tsla['Open']), 2), round(np.median(tsla['Close']), 2),
              round(np.median(tsla['Volume']), 2), round(np.median(tsla['Adj Close']), 2)]

ibm_median = ["IBM", round(np.median(ibm['High']), 2), round(np.median(ibm['Low']), 2),
             round(np.median(ibm['Open']), 2), round(np.median(ibm['Close']), 2),
             round(np.median(ibm['Volume']), 2), round(np.median(ibm['Adj Close']), 2)]

yelp_median = ["YELP", round(np.median(yelp['High']), 2), round(np.median(yelp['Low']), 2),
              round(np.median(yelp['Open']), 2), round(np.median(yelp['Close']), 2),
              round(np.median(yelp['Volume']), 2), round(np.median(yelp['Adj Close']), 2)]

msft_median = ["MSFT", round(np.median(msft['High']), 2), round(np.median(msft['Low']), 2),
              round(np.median(msft['Open']), 2), round(np.median(msft['Close']), 2),
              round(np.median(msft['Volume']), 2), round(np.median(msft['Adj Close']), 2)]

# Calculate std dev values for each ticker utilizing NumPy functions
aapl_std = ["AAPL", round(np.std(aapl['High']), 2), round(np.std(aapl['Low']), 2),
              round(np.std(aapl['Open']), 2), round(np.std(aapl['Close']), 2),
              round(np.std(aapl['Volume']), 2), round(np.std(aapl['Adj Close']), 2)]

orcl_std = ["ORCL", round(np.std(orcl['High']), 2), round(np.std(orcl['Low']), 2),
              round(np.std(orcl['Open']), 2), round(np.std(orcl['Close']), 2),
              round(np.std(orcl['Volume']), 2), round(np.std(orcl['Adj Close']), 2)]

tsla_std = ["TSLA", round(np.std(tsla['High']), 2), round(np.std(tsla['Low']), 2),
              round(np.std(tsla['Open']), 2), round(np.std(tsla['Close']), 2),
              round(np.std(tsla['Volume']), 2), round(np.std(tsla['Adj Close']), 2)]

ibm_std = ["IBM", round(np.std(ibm['High']), 2), round(np.std(ibm['Low']), 2),
             round(np.std(ibm['Open']), 2), round(np.std(ibm['Close']), 2),
             round(np.std(ibm['Volume']), 2), round(np.std(ibm['Adj Close']), 2)]

yelp_std = ["YELP", round(np.std(yelp['High']), 2), round(np.std(yelp['Low']), 2),
              round(np.std(yelp['Open']), 2), round(np.std(yelp['Close']), 2),
              round(np.std(yelp['Volume']), 2), round(np.std(yelp['Adj Close']), 2)]

msft_std = ["MSFT", round(np.std(msft['High']), 2), round(np.std(msft['Low']), 2),
              round(np.std(msft['Open']), 2), round(np.std(msft['Close']), 2),
              round(np.std(msft['Volume']), 2), round(np.std(msft['Adj Close']), 2)]

# Calculate variance values for each ticker utilizing NumPy functions
aapl_var = ["AAPL", round(np.var(aapl['High']), 2), round(np.var(aapl['Low']), 2),
              round(np.var(aapl['Open']), 2), round(np.var(aapl['Close']), 2),
              round(np.var(aapl['Volume']), 2), round(np.var(aapl['Adj Close']), 2)]

orcl_var = ["ORCL", round(np.var(orcl['High']), 2), round(np.var(orcl['Low']), 2),
              round(np.var(orcl['Open']), 2), round(np.var(orcl['Close']), 2),
              round(np.var(orcl['Volume']), 2), round(np.var(orcl['Adj Close']), 2)]

tsla_var = ["TSLA", round(np.var(tsla['High']), 2), round(np.var(tsla['Low']), 2),
              round(np.var(tsla['Open']), 2), round(np.var(tsla['Close']), 2),
              round(np.var(tsla['Volume']), 2), round(np.var(tsla['Adj Close']), 2)]

ibm_var = ["IBM", round(np.var(ibm['High']), 2), round(np.var(ibm['Low']), 2),
             round(np.var(ibm['Open']), 2), round(np.var(ibm['Close']), 2),
             round(np.var(ibm['Volume']), 2), round(np.var(ibm['Adj Close']), 2)]

yelp_var = ["YELP", round(np.var(yelp['High']), 2), round(np.var(yelp['Low']), 2),
              round(np.var(yelp['Open']), 2), round(np.var(yelp['Close']), 2),
              round(np.var(yelp['Volume']), 2), round(np.var(yelp['Adj Close']), 2)]

msft_var = ["MSFT", round(np.var(msft['High']), 2), round(np.var(msft['Low']), 2),
              round(np.var(msft['Open']), 2), round(np.var(msft['Close']), 2),
              round(np.var(msft['Volume']), 2), round(np.var(msft['Adj Close']), 2)]


#%%
# Populate empty DataFrames with calculated data above
mean_dict = {'AAPL': aapl_mean, 'ORCL': orcl_mean, 'TSLA':tsla_mean, 'IBM':ibm_mean, 'YELP':yelp_mean, 'MSFT': msft_mean}
median_dict = {'AAPL': aapl_median, 'ORCL': orcl_median, 'TSLA':tsla_median, 'IBM':ibm_median, 'YELP':yelp_median, 'MSFT': msft_median}
std_dict = {'AAPL': aapl_std, 'ORCL': orcl_std, 'TSLA':tsla_std, 'IBM':ibm_std, 'YELP':yelp_std, 'MSFT': msft_std}
var_dict = {'AAPL': aapl_var, 'ORCL': orcl_var, 'TSLA':tsla_var, 'IBM':ibm_var, 'YELP':yelp_var, 'MSFT': msft_var}

for i in stocks:
    df_mean.loc[i] = mean_dict[i]
    df_median.loc[i] = median_dict[i]
    df_std.loc[i] = std_dict[i]
    df_var.loc[i] = var_dict[i]

#%%
# Add Minimum / Maximum subtotals
outputs = [df_mean, df_median, df_var, df_std]
for df in outputs:
    df.loc['MAXIMUM'] = ['MAXIMUM', np.max(df['High']), np.max(df['Low']),
                          np.max(df['Open']), np.max(df['Close']),
                          np.max(df['Volume']), np.max(df['Adj Close']),
                          ]

    df.loc['MINIMUM'] = ['MINIMUM', np.min(df['High']), np.min(df['Low']),
                          np.min(df['Open']), np.min(df['Close']),
                          np.min(df['Volume']), np.min(df['Adj Close']),
                          ]
# NEED TO LEARN TO INCORPORATE IDX MAX - FOR FUTURE WORKFLOW
    df.loc['MAX TICKER'] = ['MAXTICKER', np.max(df['High']), np.max(df['Low']),
                            np.max(df['Open']), np.max(df['Close']),
                            np.max(df['Volume']), np.max(df['Adj Close'])
                            ]

    df.loc['MIN TICKER'] = ['MINTICKER', np.min(df['High']), np.min(df['Low']),
                            np.min(df['Open']), np.min(df['Close']),
                            np.min(df['Volume']), np.min(df['Adj Close'])
                            ]

#%%
# Reset index to display stocks / subtotals
df_mean.set_index(['Ticker'], inplace=True)
df_median.set_index(['Ticker'], inplace=True)
df_std.set_index(['Ticker'], inplace=True)
df_var.set_index(['Ticker'], inplace=True)


#%%
# GENERATE REPORT OUTPUTS
print(f'MEAN VALUES: \n{df_mean[:]}')
print(f'MEDIAN VALUES: \n{df_median[:]}')
print(f'STD DEV VALUES: \n{df_std[:]}')
print(f'VARIANCE VALUES: \n{df_var[:]}')

#%%

# QUESTION 3
# Repeat question 2 for the variance.

# Please refer to code blocks above.

# QUESTION 4
# Repeat question 2 for the std.

# Please refer to code blocks above.

# QUESTION 5
# Repeat question 2 for the median.

# Please refer to code blocks above.

#%%
# QUESTION 6
# Which company has the maximum & minimum mean in each attribute?
# Add a row to the bottom of the table and the display the table on the console.

# QUESTION 7
# Which company has the maximum & minimum variance in each attribute?
# Add a row to the bottom of the table and the display the table on the console.

# QUESTION 8
# Which company has the maximum & minimum std in each attribute?

# QUESTION 9
# Which company has the maximum & minimum median in each attribute?

# Add a row to the bottom of the table and the display the table on the console.

# please refer to abve functions incorporating 'MAXTICKER' 'MINTICKER'

#%%
# QUESTION 10

# Calculate the correlation matrix for AAPL with all the given features.
# Display the correlation matrix on the console.
# Hint. You may use .corr() for the calculation of correlation matrix.
# Write down your observation about the correlation matrix.

print(aapl.corr())

# All pricing columns are very highly correlated, though volume diverges (still strong)
# This makes sense, as generally a stock trades inside a tight band of pricing within any given day.
# Whereas, the same stock may trade at elevated or depressed volumes at much greater swings in magnitude any given day.

#%%
# QUESTION 11
# Repeat question 10 for 'ORCL', 'TSLA', 'IBM','YELP', 'MSFT'.

print(orcl.corr())
print(tsla.corr())
print(ibm.corr())
print(yelp.corr())
print(msft.corr())

# similar commentary applies as in Question 10.
# same trend between pricing can be seen across additional stocks
# certain stocks swing more wildly than others in terms of volume
# would be interesting to look at some of the retail 'meme' stocks like Gamestop or AMC


LAB #1

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1
# Using the NumPy package in python
# Number of samples for both x and y = 1000
# Create a random variable x: normally distributed about the mean of zero and variance 1
# Create a random variable y: normally distributed about the mean of 5 and variance of 2

n = 1000
meanx = 0
meany = 5
stdx = 1
stdy = 2

x = np.random.normal(meanx, stdx, n)
y = np.random.normal(meany, stdy, n)

#%%
# QUESTION 2 - MANUAL
# Write a python program that calculates Pearson's correlation coefficient
# Between two random variables x and y defined in question 1

def corr_coef_pearson(x,y):
  n = len(x)
  sum_x = float(sum(x))
  sum_y = float(sum(y))
  sum_x_sq = sum(xj*xj for xj in x)
  sum_y_sq = sum(yj*yj for yj in y)
  pearson_sum = sum(xj*yj for xj, yj in zip(x, y))
  numerator = pearson_sum - (sum_x * sum_y/n)
  denominator = pow((sum_x_sq - pow(sum_x, 2) / n) * (sum_y_sq - pow(sum_y, 2) / n), 0.5)
  if denominator == 0: return 0
  return numerator / denominator

print(f'Pearson Correlation Coefficient: {corr_coef_pearson(x,y):.5f}')

# utilized stackoverflow.com for reference in coding manual Pearson function

#%%
# QUESTION 2 - AUTOMATED
# Write a python program that calculates Pearson's correlation coefficient
# Between two random variables x and y defined in question 1

corr_coef = np.corrcoef(x, y)[0, 1]
print(corr_coef)
print(f'Pearson Correlation Coefficient: {corr_coef:.5f}')

#%%
# QUESTION 3
# Display a message on the console that shows the following information :
    # a. The sample mean of random variable x is :
    # b. The sample mean of random variable y is :
    # c. The sample variance of random variable x is :
    # d. The sample variance of random variable y is:
    # e. The sample Pearson's correlation coefficient between x & y is:

x_mean = np.mean(x)
y_mean = np.mean(y)
x_var = np.var(x)
y_var = np.var(y)

print(f'Sample Mean of Random Variable X: {x_mean:.2f}')
print(f'Sample Mean of Random Variable Y: {y_mean:.2f}')
print(f'Sample Variance of Random Variable X: {x_var:.2f}')
print(f'Sample Variance of Random Variable Y: {y_var:.2f}')
print(f'Sample Pearson Correlation Coefficient Between X+Y: {corr_coef:.5f}')

#%%
# QUESTION 4
# Using the matplotlib.pyplot package in python:
    # Display the line plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.
    # Hint: You need to use plt.plot()

plt.figure(figsize=(12,12))
plt.plot(x, label='X Variables')
plt.plot(y, label='Y Variables')
plt.title(f'LINE PLOT OF X+Y:', fontsize=20)
plt.xlabel('SAMPLE POINTS (#)', fontsize=20)
plt.ylabel('VALUE (#)', fontsize=20)
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
# QUESTION 5
# Using the matplotlib.pyplot package in python:
    # Display the histogram plot of the random variable x and y in one figure differentiating x and y with legend
    # Add an appropriate x-label, y-label, title, and legend to each graph.

plt.figure(figsize=(12,12))
plt.hist(x, bins=50, label='X Variables', orientation='vertical', alpha=0.5)
plt.hist(y, bins=50, label='Y Variables', orientation='vertical', alpha=0.5)
plt.title(f'HISTOGRAM PLOT OF X+Y:', fontsize=20)
plt.xlabel('VALUE (#)', fontsize=20)
plt.ylabel('FREQUENCY (#)', fontsize=20)
plt.legend()
plt.grid()
plt.show()


#%%
# QUESTION 6
# Using pandas package in python read in the 'tute1.csv' dataset
    # Timeseries dataset with Sales, AdBudget and GDP column

df = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/tute1.csv', index_col=0)
print(df.head())
print(df.info())

#print(df.describe())
#col_names = df.columns
#print(col_names)

#%%
# QUESTION 7
# Find the Pearson's correlation coefficient between:
    # a. Sales & AdBudget
    # b. Sales & GDP
    # c. AdBudget & GDP

sales_ads_corr_coef = np.corrcoef(df['Sales'], df['AdBudget'])[0,1]
sales_gdp_corr_coef = np.corrcoef(df['Sales'], df['GDP'])[0,1]
ads_gdp_corr_coef = np.corrcoef(df['AdBudget'], df['GDP'])[0,1]

#%%
# QUESTION 8
# Display a message on the console that shows the following:
    # a. The sample Pearson's correlation coefficient between Sales & AdBudget is:
    # b. The sample Pearson's correlation coefficient between Sales & GDP is:
    # c. The sample Pearson's correlation coefficient between AdBudget & GDP is:

print(f'Sample Pearson Correlation Coefficient between Sales & AdBudget: {sales_ads_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between Sales & GDP: {sales_gdp_corr_coef:.5f}')
print(f'Sample Pearson Correlation Coefficient between AdBudget & GDP: {ads_gdp_corr_coef:.5f}')

#%%
# QUESTION 9
# Display the line plot of Sales, AdBudget and GDP in one graph versus time
# Add an appropriate x- label, y-label, title, and legend to each graph.
# Hint: You need to us the plt.plot().

plt.figure(figsize=(12,12))
plt.plot(df['Sales'], 'b', label='Sales')
plt.plot(df['AdBudget'], 'r', label='AdBudget')
plt.plot(df['GDP'], 'g', label='GDP')
plt.title(f'LINE PLOT OF SALES / AD BUDGET / GDP:', fontsize=20)
plt.xlabel('DATE / TIME', fontsize=20)
plt.ylabel('VALUE ($)', fontsize=20)
plt.legend(loc='best')
plt.grid()
plt.show()


#%%
# QUESTION 10
# Plot the histogram plot of Sales, AdBudget and GDP in one graph.
# Add an appropriate x-label, y- label, title, and legend to each graph.
# Hint: You need to us the plt.hist().

plt.figure(figsize=(12,12))
plt.hist(df['Sales'], color='b', bins=50, label='Sales', orientation='vertical', alpha=0.5)
plt.hist(df['AdBudget'], color='r', bins=50, label='AdBudget', orientation='vertical', alpha=0.5)
plt.hist(df['GDP'], color='g', bins=50, label='GDP', orientation='vertical', alpha=0.5)
plt.title(f'HISTOGRAM PLOT OF SALES / AD BUDGET / GDP', fontsize=20)
plt.xlabel('VALUE ($)', fontsize=20)
plt.ylabel('FREQUENCY (#)', fontsize=20)
plt.legend()
plt.grid()
plt.show()




HW #1

#%%
# LIBRARY IMPORTS
import numpy as np
import pandas as pd
import pandas_datareader as web

#import matplotlib.pyplot as plt
#from tabulate import tabulate

# import seaborn as sns
# from scipy import stats as stats
# import statistics

print("\nIMPORT SUCCESS")

#%%
# QUESTION 1
# Using the python pandas package load the titanic.csv dataset.
# Write a python program that reads all the column title and save them under variable 'col'.
# The list of features in the dataset and the explanation is as followed:

titanic = pd.read_csv('/Users/nehat312/GitHub/Complex-Data-Visualization-/titanic.csv')
col = titanic.columns
print(titanic.info())
print(col)

#%%
# QUESTION 2
# The titanic dataset needs to be cleaned due to nan entries.
# Remove all the nan in the dataset.
# Using the .describe() and .head() display the cleaned data.

titanic_drop = titanic.dropna(axis=0)
print(titanic_drop.head())
print(titanic_drop.describe())

#%%
# QUESTION 3
# The titanic dataset needs to be cleaned due to nan entries.
# Replace all the nan in the dataset with the mean of the column.
# Using the .describe() and .head() display the cleaned data.

titanic_mean = titanic.fillna(titanic.mean())
print(titanic_mean.head())
print(titanic_mean.describe())

#%%
# QUESTION 4
# The titanic dataset needs to be cleaned due to nan entries.
# Replace all the nan in the dataset with the median of the column.
# Using the .describe() and .head() display the cleaned data.

titanic_median = titanic.fillna(titanic.median())
print(titanic_median.head())
print(titanic_median.describe())

#%%
# QUESTION 5
# Using pandas in python write a program that finds the total number of passengers on board.
total_passengers = sum((titanic.Age > 0))
titanic_males = titanic[titanic['Sex'] == 'male']
titanic_females = titanic[titanic['Sex'] == 'female']

# Then finds the number of males and females on board and display the following message:
# a. The total number of passengers on board was
print(f'The total number of passengers on board was {len(titanic)}.')

# b. From the total number of passengers onboard, there were % male passengers onboard.
print(f'From the total number of passengers onboard, there were {100*(sum(titanic_males.Age>0)/len(titanic)):.2f} % male passengers onboard.') # / total_passengers

# From the total number of passengers onboard, there were % female passengers # onboard.
    # (Display the percentage value here with 2-digit precision)
print(f'From the total number of passengers onboard, there were {100*(1-(sum(titanic_males.Age>0)/len(titanic))):.2f} % female passengers onboard.')

# d. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 6
# Using pandas in python write a program that find out the number of survivals.
# Also, the total number of male survivals and total number of female survivals.
    # Note: You need to use the Boolean & condition to filter the DataFrame.

# Then display a message on the console as follows:
# a. Total number of survivals onboard was .
titanic_survived = titanic[(titanic['Survived'] == 1)]
print(f'Total number of survivals onboard was {sum(titanic_survived.Age>0):.0f}.')

# b. Out of male passengers onboard only % male was survived.
    # (Display the percentage value here with 2-digit precision)
titanic_males_survived = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'male')]
print(f'Out of male passengers onboard only {100*(sum(titanic_males_survived.Age>0)/len(titanic)):.2f} % male was survived.')

# c. Out of female passengers onboard only % female was survived.
    # (Display the percentage value here with 2-digit precision).
titanic_females_survived = titanic[(titanic['Survived'] == 1) & (titanic['Sex'] == 'female')]
print(f'Out of female passengers onboard only {100*(sum(titanic_females_survived.Age>0)/len(titanic)):.2f} % female was survived.')

# d. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 7
# Using pandas in python write a program that find out the number of passengers with upper class ticket.
# Then, find out the total number of male and female survivals with upper class ticket.
# Then display the following messages on the console:

# a. There were total number of passengers with the upper-class ticket and only % were survived.
    # (Display the percentage value here with 2-digit precision).

# b. Out of passengers with upper class ticket, % passengers were male.
    # (Display the percentage value here with 2-digit precision).

# c. Out of passengers with upper class ticket, % passengers were female.
    # (Display the percentage value here with 2-digit precision).

# d. Out of passengers with upper class ticket, % male passengers were survived.
    # (Display the percentage value here with 2-digit precision).
# e. Out of passengers with upper class ticket, % female passengers were survived.
    # (Display the percentage value here with 2-digit precision).
# f. Print some dash lines here to separate the displayed message from the next question.
print(f'--------------------------------------------------------')

#%%
# QUESTION 8
# Using pandas & numpy package add a column to the dataset called "Above50&Male".
# Entries corresponding to this column must display "Yes" if a passenger is Male & more than 50 yo, otherwise "No".
# Display the first 5 rows of the new dataset with the added column.
# How many male passengers onboard above 50 years old? Display the information on the console.

titanic['Above50Male'] = np.where((titanic['Age'] > 50) & (titanic['Sex'] == 'male'), 'Yes', 'No')
male50plus = sum((titanic.Age > 50) & (titanic.Sex == 'male'))
print(titanic.head())
print('------------------------------------------------------')
print(f'Male Passengers Over 50 Years Old: {male50plus}')

#%%
# QUESTION 9

# Using pandas & numpy package add a column to the dataset called "Above50&Male&Survived".
# Entries corresponding to this column must display "Yes" if a passenger is Male & more than 50 yo & survived, otherwise "No".
# Display the first 5 rows of the new dataset with the added column.
# How many male passengers onboard were above 50 & survived? Display the information on the console.
# Find the survival percentage rate of male passengers onboard who are above 50 years old?

titanic['Above50Male_Survived'] = np.where((titanic['Age'] >= 51) & (titanic['Sex'] == 'male') & (titanic['Survived'] == 1), 'Yes', 'No')
male50plus_survived = sum((titanic.Age > 50) & (titanic.Sex == 'male') & (titanic.Survived == 1))
print(titanic.head())
print('------------------------------------------------------')
print(f'Male Survivors Over 50 Years Old: {male50plus_survived}')

#%%
# QUESTION 10
# Repeat question 8 & 9 for the female passengers.

titanic['Above50Female'] = np.where((titanic['Age'] > 50) & (titanic['Sex'] == 'female'), 'Yes', 'No')
female50plus = sum((titanic.Age > 50) & (titanic.Sex == 'female'))
print(titanic.head())
print('------------------------------------------------------')
print(f'Female Passengers Over 50 Years Old: {female50plus}')

titanic['Above50Female_Survived'] = np.where((titanic['Age'] >= 51) & (titanic['Sex'] == 'female') & (titanic['Survived'] == 1), 'Yes', 'No')
female50plus_survived = sum((titanic.Age > 50) & (titanic.Sex == 'female') & (titanic.Survived == 1))
print(titanic.head())
print('------------------------------------------------------')
print(f'Female Survivors Over 50 Years Old: {female50plus_survived}')


#%%

# SCRATCH NOTES
survived_status = titanic.iloc[:,0]
print(survived_status)

titanic['age_over_mean'] = titanic['age'] / np.mean(titanic['age'])
print(titanic.head())
#titanic.insert(loc=4, column='age_over_mean')




DATETIME CONVERSION / MAPPING

mo_abr_map = {'Jan': '1', 'Feb': '1', 'Mar': '3',
              'Apr': '4', 'May': '5', 'Jun': '6',
              'Jul': '7', 'Aug': '8', 'Sep': '9',
              'Oct': '10', 'Nov': '11', 'Dec': '12'}

mo_day_map = {'1': '31', '2': '28', '3': '31',
              '4': '30', '5': '31', '6': '30',
              '7': '31', '8': '31', '9': '30',
              '10': '31', '11': '30', '12': '31'}

mo_num_qtr_map = {'01': '1', '02': '1', '03': '1',
              '04': '2', '05': '2', '06': '2',
              '07': '3', '08': '3', '09': '3',
              '10': '4', '11': '4', '12': '4'}

mo_abr_qtr_map = {'Jan': '1', 'Feb': '1', 'Mar': '1',
              'Apr': '2', 'May': '2', 'Jun': '2',
              'Jul': '3', 'Aug': '3', 'Sep': '3',
              'Oct': '4', 'Nov': '4', 'Dec': '4'}


