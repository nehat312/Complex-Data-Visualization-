#%% [markdown]
# DATS-6401 - CLASS 2/9/22
# Nate Ehat

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
