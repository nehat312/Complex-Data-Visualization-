
import dash
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dash import html
from dash import dcc
from dash.dependencies import Input,Output
import plotly.express as px
from scipy.fft import fft
from dash.exceptions import PreventUpdate
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot

#1
url='https://raw.githubusercontent.com/rjafari979/Complex-Data-Visualization-/main/Metro_Interstate_Traffic_Volume.csv'
df=pd.read_csv(url)
print(df.head(2))
sns.set_style('whitegrid')
sns.set_style(style='ticks')
print(df.describe())

#2
print(df.isnull().sum().sort_values(ascending=False))
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
categorical_features = df.select_dtypes(include=['object']).columns

#3
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

X=StandardScaler().fit_transform(df[df.columns[1:5]].values)
H_pca = np.matmul(df[df.columns[1:5]].values.T,df[df.columns[1:5]].values)
_,d_PCA,_ = np.linalg.svd(H_pca)
print(f'Transformed X singular values: {d_PCA}')
print(f'Transformed X condition number: {np.linalg.cond(df[df.columns[1:5]].values)}')
df1=df[df.columns[1:5]]
sns.heatmap(df1.corr(),annot=True)
plt.title("correlation coefficient between features - orginal feature space")
plt.show()
pca = PCA(n_components='mle', svd_solver='full')
pca.fit(X)
X_pca = pca.transform(X)
print("explained variance ratio:",pca.explained_variance_ratio_)
print("shape:",X_pca.shape)
'''
3 features should be removed because explained variance ratio is greater than 0.01
'''
plt.plot(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1,1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(np.arange(1, len(np.cumsum(pca.explained_variance_ratio_))+1,1))
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance")
plt.show()
df2=pd.DataFrame(X_pca).corr()
a,b = X_pca.shape
col = []
for i in range(b):
    col.append(f'Principle column {i+1}')
sns.heatmap(df2,annot=True,xticklabels=col,yticklabels=col)
plt.title("correlation coefficient")
plt.show()
df_PCA = pd.DataFrame(data=X_pca, columns=col)
print("old one:",df1.head().to_string)
print("new one:",df_PCA.head().to_string)

#4

q1_tv, q2_tv, q3_tv = df['traffic_volume'].quantile([0.25,0.5,0.75])
iqr_tv = q3_tv - q1_tv
lower_limit_tv = q1_tv - 1.5*iqr_tv
upper_limit_tv = q3_tv + 1.5*iqr_tv
print(f'Q1 and Q3 of the traffic_volume is {q1_tv}  & {q3_tv:} ')
print(f'IQR for the traffic_volume is {iqr_tv:} ')
print(f'Any traffic_volume < {lower_limit_tv}and traffic_volume > {upper_limit_tv} is an outlier')
sns.set_style('darkgrid')
sns.boxplot(y=df['traffic_volume'])
plt.title('Boxplot - traffic_volume')
plt.show()

df_clean = df[(df['traffic_volume'] >= lower_limit_tv) & (df['traffic_volume'] <= upper_limit_tv)]
sns.set_style('darkgrid')
sns.boxplot(y=df_clean['traffic_volume'])
plt.title('Boxplot of traffic_volume after removing outliers')
plt.show()

#5
q1_t, q2_t, q3_t = df['temp'].quantile([0.25,0.5,0.75])
iqr_t = q3_t - q1_t
lower_limit_t = q1_t - 1.5*iqr_t
upper_limit_t = q3_t + 1.5*iqr_t
print(f'Q1 and Q3 of the temp is {q1_t}  & {q3_t:} ')
print(f'IQR for the temp is {iqr_t:} ')
print(f'Any temp < {lower_limit_t}and temp > {upper_limit_t} is an outlier')
sns.set_style('darkgrid')
sns.boxplot(y=df['temp'])
plt.title('Boxplot - temp')
plt.show()

df_clean1 = df[(df['temp'] >= lower_limit_t) & (df['temp'] <= upper_limit_t)]
sns.set_style('darkgrid')
sns.boxplot(y=df_clean1['temp'])
plt.title('Boxplot of temp after removing outliers')
plt.show()

#6
sns.heatmap(df.corr(),annot=True).set(xlabel='traffic volume',ylabel='features')
plt.title("heatmap")
plt.show()
#7
sns.barplot(data=df,y="traffic_volume",x='weather_main').set(xlabel='weather type',ylabel='traffic volume')
plt.title("barplot-traffic volume vs weather type")
plt.show()
#8
sns.histplot(data=df,x='traffic_volume',element='poly',hue='weather_main').set(xlabel='traffic volume',ylabel='count')
plt.title("histplot-traffic vol (poly)")

plt.show()
#9
sns.histplot(data=df,x='traffic_volume',element='bars',hue='weather_main').set(xlabel='traffic volume',ylabel='count')
plt.title("histplot-traffic vol (bars)")

plt.show()
#10
sns.histplot(data=df,x='traffic_volume',element='step',hue='weather_main').set(xlabel='traffic volume',ylabel='count')
plt.title("histplot-traffic vol (step)")
plt.legend('weather_main')
plt.show()
#11
sns.histplot(data=df,x='traffic_volume',multiple='stack',hue='weather_main').set(xlabel='traffic volume',ylabel='count')
plt.title("histplot-traffic vol (stack)")

plt.show()

#12
sns.kdeplot(data=df,x='traffic_volume',hue='weather_main').set(xlabel='traffic volume',ylabel='count')
plt.title("kdeplot-traffic vol ")

plt.show()
#13

df_z = pd.DataFrame()
df_z['z_tv'] = st.zscore(df['traffic_volume'])
df_z['z_t'] = st.zscore(df['temp'])

sns.set_style('darkgrid')
sns.lineplot(data=df_z[:100]).set(xlabel= 'Observations',ylabel='z-score')
plt.title('z-score of tv and t vs Number of Observations ')
plt.show()

print("Sample mean for z_score: ", df_z[:100].mean())
print('Sample variance for z_score:',df_z[:100].var())

# 4
sns.set_style('darkgrid')
sns.histplot(data=df_z,bins=50).set(xlabel='z-score')
plt.title('Histogram of z-scores of tv and t ')
plt.show()

# 5
fig, axes = plt.subplots(2,2,figsize=(20,20))
sns.set_style('darkgrid')
sns.lineplot(data=df[:100],ax=axes[0,0]).set(xlabel= 'Observations',ylabel='Values')
axes[0,0].set_title('traffic vol and temp vs Number of Observations ')

sns.set_style('darkgrid')
sns.histplot(data=df,bins=50,ax=axes[1,0]).set(xlabel='Values')
axes[1,0].set_title('Histogram of traffic vol and temp ')

sns.set_style('darkgrid')
sns.lineplot(data=df_z[:100],ax=axes[0,1]).set(xlabel= 'Observations',ylabel='z-score')
axes[0,1].set_title('z-score of traffic vol and temp vs Number of Observations ')

sns.set_style('darkgrid')
sns.histplot(data=df_z,bins=50,ax=axes[1,1]).set(xlabel='z-score')
axes[1,1].set_title('Histogram of z-scores of traffic vol and temp ')
plt.show()

#14
kstest_tv = st.kstest(df['traffic_volume'],'norm')
kstest_t = st.kstest(df['temp'],'norm')
da_testtv = st.normaltest(df['traffic_volume'])
da_testt = st.normaltest(df['temp'])
shapiro_testtv = st.shapiro(df['traffic_volume'])
shapiro_testt = st.shapiro(df['temp'])
import dash_bootstrap_components as dbc
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = html.Div([html.H1('test', style={'textAlign':'center'}),
                         dcc.Graph(id="my_graph"),
                         dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'dependent variable', 'value': 'var'},
                {'label': 'independent variable', 'value': 'var2'},
                    ],
            value='var'),
            html.H1(["choose option:"]),
            dcc.Dropdown(
            id='con_dropdown',
            options=[
                {'label': 'Da_k_squared', 'value': 'Da_k_squared'},
                {'label': 'K_S test', 'value': 'K_S test'},
                {'label': 'Shapiro Test', 'value': 'Shapiro Test'},

                    ],
            value='Da_k_squared',

            ),




 ])

@app.callback(
     Output(component_id='my_graph', component_property='figure'),
     [Input(component_id='dropdown', component_property='value'),
      Input(component_id='con_dropdown',component_property='value')]

)
def select_graph(value1, value2):
        if value1 == 'var':
            if value2=='K_S test':
                print(f"K-S test: statistics={kstest_tv[0]}, p-value={kstest_tv[1]}")
            elif value2=='Da_k_squared':
                print(f"da_k_squared test: statistics={da_testtv[0]:.5f}, p-value={da_testtv[1]:.5f}")
            elif value2 == 'Shapiro Test':
                print(f"Shapiro test: statistics={shapiro_testtv[0]:.5f}, p-value={shapiro_testtv[1]:.5f}")

        elif value1=='var2':
            if value2=='K_S test':
                print(f"K-S test: statistics={kstest_t[0]}, p-value={kstest_t[1]}")
            elif value2=='Da_k_squared':
                print(f"da_k_squared test: statistics={da_testt[0]:.5f}, p-value={da_testt[1]:.5f}")
            elif value2 == 'Shapiro Test':
                print(f"Shapiro test: statistics={shapiro_testt[0]:.5f}, p-value={shapiro_testt[1]:.5f}")


app.run_server(debug=True, port=3001)

#15
qqplot(df['traffic_volume'])
plt.title("QQ-plot of traffic volume ")
plt.show()

# 16
qqplot(df['temp'])
plt.title("QQ-plot of temp")
plt.show()














