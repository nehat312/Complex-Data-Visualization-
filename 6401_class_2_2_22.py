#%% [markdown]
# DATS-6401 - CLASS 2-2-22
# Nate Ehat

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

