import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
from statsmodels.graphics.gofplots import qqplot

# 1
x = np.random.randn(5000)
y = np.cumsum(x)

fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000),y=x,ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000),y=y,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.histplot(x=x,bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=y,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram Non-Gaussian data')
plt.tight_layout()
plt.show()

# 2
kstest_x = st.kstest(x,'norm')
kstest_y = st.kstest(y,'norm')

print(f"K-S test: statistics={kstest_x[0]:.5f}, p-value={kstest_x[1]:.5f}")
print(f"K-S test: x dataset looks {'Normal' if kstest_x[1] > 0.01 else 'Non-Normal'}")
print(f"K-S test: statistics={kstest_y[0]:.5f}, p-value={kstest_y[1]:.5f}")
print(f"K-S test: y dataset looks {'Normal' if kstest_y[1] > 0.01 else 'Non-Normal'}")

# 3
shapiro_test_x = st.shapiro(x)
shapiro_test_y = st.shapiro(y)

print(f"Shapiro test: statistics={shapiro_test_x[0]:.5f}, p-value={shapiro_test_x[1]:.5f}")
print(f"Shapiro test: x dataset looks {'Normal' if shapiro_test_x[1] > 0.01 else 'Non-Normal'}")
print(f"Shapiro test: statistics={shapiro_test_y[0]:.5f}, p-value={shapiro_test_y[1]:.5f}")
print(f"Shapiro test: y dataset looks {'Normal' if shapiro_test_y[1] > 0.01 else 'Non-Normal'}")

# 4
da_test_x = st.normaltest(x)
da_test_y = st.normaltest(y)

print(f"da_k_squared test: statistics={da_test_x[0]:.5f}, p-value={da_test_x[1]:.5f}")
print(f"da_k_squared test: x dataset looks {'Normal' if da_test_x[1] > 0.01 else 'Non-Normal'}")
print(f"da_k_squared test: statistics={da_test_y[0]:.5f}, p-value={da_test_y[1]:.5f}")
print(f"da_k_squared test: y dataset looks {'Normal' if da_test_y[1] > 0.01 else 'Non-Normal'}")

# 5
new_y = st.norm.ppf(st.rankdata(y)/(len(y) + 1))
fig, axes = plt.subplots(2,2,figsize=(9,7))
sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000),y=y,ax=axes[0,0]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,0].set_title('Non-Gaussian data')

sns.set_style('darkgrid')
sns.lineplot(x=np.linspace(1,5000,5000),y=new_y,ax=axes[0,1]).set(xlabel= '# of samples',ylabel='Magnitude')
axes[0,1].set_title('Transformed data (Gaussian)')

sns.set_style('darkgrid')
sns.histplot(x=y,bins=100,ax=axes[1,0]).set(xlabel='Magnitude')
axes[1,0].set_title('Histogram of Non-Gaussian Data')

sns.set_style('darkgrid')
sns.histplot(x=new_y,bins=100,ax=axes[1,1]).set(xlabel='Magnitude')
axes[1,1].set_title('Histogram of Transformed data (Gaussian)')
plt.tight_layout()
plt.show()

# 6
fig, ax = plt.subplots(ncols=2)
qqplot(y,ax=ax[0])
ax[0].set_title('y Data: Non-Normal')
qqplot(new_y,ax=ax[1])
ax[1].set_title('transformed y:Normal')
plt.show()

# 7
kstest_new_y = st.kstest(new_y,'norm')

print(f"K-S test: statistics={kstest_new_y[0]:.5f}, p-value={kstest_new_y[1]:.5f}")
print(f"K-S test: new_y dataset looks {'Normal' if kstest_new_y[1] > 0.01 else 'Non-Normal'}")

# 8
shapiro_test_new_y = st.shapiro(new_y)

print(f"Shapiro test: statistics={shapiro_test_new_y[0]:.5f}, p-value={shapiro_test_new_y[1]:.5f}")
print(f"Shapiro test: new_y dataset looks {'Normal' if shapiro_test_new_y[1] > 0.01 else 'Non-Normal'}")

# 9
da_test_new_y = st.normaltest(new_y)

print(f"da_k_squared test: statistics={da_test_new_y[0]:.5f}, p-value={da_test_new_y[1]:.5f}")
print(f"da_k_squared test: new_y dataset looks {'Normal' if da_test_new_y[1] > 0.01 else 'Non-Normal'}")
