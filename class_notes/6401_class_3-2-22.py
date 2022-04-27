#%% [markdown]
# DATS-6401 - CLASS 3/2/22
# Nate Ehat

#%%
# LIBRARY IMPORTS

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly as ply
import plotly.express as px
import pandas_datareader as web

from statsmodels.graphics.gofplots import qqplot

import seaborn as sns
# from scipy import stats as stats
# import statistics
# import datetime as dt

print("\nIMPORT SUCCESS")

#%%

fig=px.line(x=[1,2,3], y=[1,2,3])
fig.show(rendered = 'browser')


#%%
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']
columns=['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
start_date = '2016-01-01'
end_date = '2022-03-02'

print("\nVARIABLES ASSIGNED")

# Pull ticker data
aapl = web.DataReader('AAPL', data_source='yahoo', start=start_date, end=end_date)
orcl = web.DataReader('ORCL', data_source='yahoo', start=start_date, end=end_date)
tsla = web.DataReader('TSLA', data_source='yahoo', start=start_date, end=end_date)
ibm = web.DataReader('IBM', data_source='yahoo', start=start_date, end=end_date)
yelp = web.DataReader('YELP', data_source='yahoo', start=start_date, end=end_date)
msft = web.DataReader('MSFT', data_source='yahoo', start=start_date, end=end_date)

df1 = pd.DataFrame(data=aapl)
df1['company'] = 'AAPL'

df2 = pd.DataFrame(data=tsla)
df2['company'] = 'TSLA'

df3 = pd.DataFrame(data=orcl)
df3['company'] = 'ORCL'

df4 = pd.DataFrame(data=ibm)
df4['company'] = 'IBM'

df5 = pd.DataFrame(data=yelp)
df5['company'] = 'YELP'

df6 = pd.DataFrame(data=msft)
df6['company'] = 'MSFT'

stock_pulls = [aapl, orcl, tsla, ibm, yelp, msft]

frames = [df1,df2,df3,df4,df5,df6]
results = pd.concat(frames)

print("\nSTOCKS PULLED")

#%%
fig=px.line(x=tsla.index, y=tsla.Close)
fig.show(rendered = 'browser')

#%%
print(results)

#%%
fig=px.line(data_frame = results, x=results.index, y='company')
fig.show(rendered = 'browser')

#%%

from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig = make_subplots(rows=2, cols=3,
                    subplot_titles=('TSLA', 'AAPL', 'IBM', 'ORCL', 'YELP', 'MSFT'))

fig.add_trace(go.Scatter(x=tsla.index, y=tsla.Close),
    row=1, col=1, name="TSLA",
)

fig.add_trace(go.Scatter(x=aapl.index, y=aapl.Close),
    row=1, col=2, name="AAPL",
)

fig.add_trace(go.Scatter(x=ibm.index, y=ibm.Close),
    row=1, col=3, name="IBM",
)

fig.add_trace(go.Scatter(x=orcl.index, y=orcl.Close),
    row=2, col=1, name="ORCL",
)

fig.add_trace(go.Scatter(x=yelp.index, y=yelp.Close),
    row=2, col=2, name="YELP",
)

fig.add_trace(go.Scatter(x=msft.index, y=msft.Close),
    row=2, col=3, name="MSFT",
)

fig.update_layout(height=600, width=800, title_text="STOCKZZZ")
fig.show(rendered = 'browser')

#%%

result = stock_pulls
result.head()

#%%
iris = sns.load_dataset('iris')
iris_cols = iris.columns
print(iris.info())
print('---------------------------------------------------------------------')
print(iris.describe())
print('---------------------------------------------------------------------')
print(f'MISSING/NULL VALUES:')
print(iris.isnull().sum())


#%%
fig=px.bar(iris, x='sepal_width', y='sepal_length')
fig.show(rendered = 'browser')

#%%

fig=px.bar(iris, x='sepal_width', y='sepal_length',
           color='species', hover_data=['petal_width', 'petal_length'])
fig.show(rendered = 'browser')

#%%
fig=px.bar(iris, x='sepal_width', y='sepal_length',
           orientation='h', color = 'species', barmode='stack',
           hover_data=['petal_width', 'petal_length'])
fig.show(rendered = 'browser')

#%%

#%%
tips = sns.load_dataset('tips')
tips_cols = tips.columns
print(tips.info())
print('---------------------------------------------------------------------')
print(tips.describe())
print('---------------------------------------------------------------------')
print(f'MISSING/NULL VALUES:')
print(tips.isnull().sum())

#%%
fig=px.bar(tips, x='day',
           color = 'smoker',  orientation='h',
           hover_data=['time'],
           ).update_xaxes(categoryorder='total ascending')
fig.show(rendered = 'browser')

#%%

fig=px.bar(tips, x='day',
           color = 'smoker',  orientation='h',
           hover_data=['time'],
           ).update_xaxes(categoryorder='total ascending')
fig.show(rendered = 'browser')

#%%
iris_cols = iris.columns
print(iris_cols)

#%%

fig=px.scatter_matrix(iris, iris_cols, color='species',)
fig.update_traces(diagonal_visible=False)
fig.show(rendered = 'browser')


#%%
fig=px.histogram(iris, x='sepal_length', y='petal_width', nbins=50, color='species',)
fig.show(rendered = 'browser')

#%%
fig=px.histogram(tips, x='total_bill', color='smoker', nbins=50, marginal='rug') #marginal='violin'
fig.show(rendered = 'browser')


#%%
import plotly.graph_objects
from plotly.subplots import make_subplots

fig = go.Figure(data=[go.Histogram(x=iris['sepal_width'], nbinsx = 50)],
                )
fig.show(rendered = 'browser')

#%%
fig = go.Figure()
fig.add_trace(go.Histogram(x=iris['sepal_width'], nbinsx=50))
fig.add_trace(go.Histogram(x=iris['sepal_length'],nbinsx=50))
fig.update_layout(barmode='stack')
fig.show(rendered = 'browser')

#%%
fig = go.Figure()
fig.add_trace(go.Scatter(x=iris['sepal_width']))
fig.add_trace(go.Scatter(x=iris['sepal_length']))
fig.update_layout(barmode='stack')
fig.show(rendered = 'browser')

#%%
tips = px.data.tips()

fig = px.pie(tips,
             values = 'total_bill',
             names='day',
             )
fig.show(rendered = 'browser')

#%%
fig = px.box(tips,
             x='day',
             y='total_bill',
             )
fig.show(rendered = 'browser')
#%%
fig = px.violin(tips,
             x='day',
             y='total_bill',
             )
fig.show(rendered = 'browser')
#%%
fig = make_subplots(rows=2, cols=2)
fig.add_trace(
    go.Scatter(x=[1,2,3], y=[5,6,7]),
    row = 1, col = 1,
)

fig.add_trace(
    go.Scatter(x=[1,2,3], y=[5,6,7]),
    row = 1, col = 2,
)

fig.add_trace(
    go.Scatter(x=[1,2,3], y=[5,6,7]),
    row = 2, col = 1,
)

fig.add_trace(
    go.Scatter(x=[1,2,3], y=[5,6,7]),
    row = 2, col = 2,
)

fig.show(rendered = 'browser')

#%%
fig = make_subplots(rows=2, cols=3,
                    subplot_titles=('TSLA', 'AAPL', 'IBM', 'ORCL', 'YELP', 'MSFT'))

fig.add_trace(go.Scatter(x=tsla.index, y=tsla.Volume),
    row=1, col=1,
)

fig.add_trace(go.Scatter(x=aapl.index, y=aapl.Volume),
    row=1, col=2,
)

fig.add_trace(go.Scatter(x=ibm.index, y=ibm.Volume),
    row=1, col=3,
)

fig.add_trace(go.Scatter(x=orcl.index, y=orcl.Volume),
    row=2, col=1,
)

fig.add_trace(go.Scatter(x=yelp.index, y=yelp.Volume),
    row=2, col=2,
)

fig.add_trace(go.Scatter(x=msft.index, y=msft.Volume),
    row=2, col=3,
)

fig.update_layout(height=600, width=800, title_text="STOCK VOLUME")
fig.show(rendered = 'browser')

#%%
fig = make_subplots(rows=2, cols=3,
                    subplot_titles=('TSLA', 'AAPL', 'IBM', 'ORCL', 'YELP', 'MSFT'))

fig.add_trace(go.Scatter(x=tsla.index, y=tsla.Volume),
    row=1, col=1,
)

fig.add_trace(go.Scatter(x=aapl.index, y=aapl.Volume),
    row=1, col=2,
)

fig.add_trace(go.Scatter(x=ibm.index, y=ibm.Volume),
    row=1, col=3,
)

fig.add_trace(go.Scatter(x=orcl.index, y=orcl.Volume),
    row=2, col=1,
)

fig.add_trace(go.Scatter(x=yelp.index, y=yelp.Volume),
    row=2, col=2,
)

fig.add_trace(go.Scatter(x=msft.index, y=msft.Volume),
    row=2, col=3,
)

fig.update_layout(height=600, width=800, title_text="STOCK VOLUME")
fig.show(rendered = 'browser')

#%%
#%%
fig = make_subplots(rows=1, cols=2,
                    subplot_titles=('TSLA', 'AAPL'),
                    specs=[[{'type':'pie'},{'type':'bar'}]])

fig.add_trace(go.Pie(values=results.Volume,
                     labels=results.company),
              row=1, col=1,
)

fig.add_trace(go.Bar(x=results.company,
                     y=results.Volume),
              row=1, col=2,
)

fig.show(rendered = 'browser')

#%%

#%%