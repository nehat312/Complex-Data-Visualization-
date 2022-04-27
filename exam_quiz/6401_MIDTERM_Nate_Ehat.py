#%% [markdown]
# DATS-6401 - MIDTERM
# Nate Ehat

#%%
# LIBRARY IMPORTS

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd

from scipy import stats as stats
import statistics

print("\nIMPORT SUCCESS")

#%%
# 1.	Using the seaborn package load the ‘diamonds’ dataset.
# Check if the dataset contains missing values or ‘nan’s.
# If it does then remove all the missing observations.
# All the questions bellow must be answered using the ‘clean’ dataset.
# Set the matplotlib style as ‘fivethirtyeight’. [5pts]

plt.style.use('fivethirtyeight')
diamond = sns.load_dataset('diamonds')
diamond_cols = diamond.columns
print(diamond.info())
print('---------------------------------------------------------------------')
print(diamond.describe())
print('---------------------------------------------------------------------')
print(f'MISSING/NULL VALUES:')
print(diamond.isnull().sum())

# NO MISSING / NULL VALUES IDENTIFIED - proceeding with original clean dataset

#%%
# 2.	Write a python program that displays the various type of diamonds ‘cut’ on the console. [5pts]
# The list of diamond cuts in the diamond dataset are: _____

cut_types = list(diamond['cut'].unique())
print(f'UNIQUE DIAMOND CUT COUNT: {len(cut_types)} CUTS')
print(f'UNIQUE DIAMOND CUT TYPES:{cut_types}')

#%%
# 3.	Write a python program that displays the various type of diamonds ‘color’ on the console. [5pts]
# The list of diamond color in the diamond dataset are: _____

color_types = list(diamond['color'].unique())
print(f'UNIQUE DIAMOND COLOR COUNT: {len(color_types)} COLORS')
print(f'UNIQUE DIAMOND COLOR TYPES:{color_types}')

#%%
# 4.	Write a python program that displays the various type of diamonds ‘clarity’ on the console. [5pts]
# The list of diamond clarity in the diamond dataset are: _____

clarity_types = list(diamond['clarity'].unique())
print(f'UNIQUE DIAMOND CLARITY COUNT: {len(clarity_types)} CLARITY')
print(f'UNIQUE DIAMOND CLARITY TYPES:{clarity_types}')

#%% [markdown]
# 5. Graph a horizontal bar-type plot that shows the diamond ‘cut’ versus the sales.
# The y-axis should display the various diamond ‘cut’
# the x-axis should display the sales count per category.
# Add the title under ‘Sales count per cut’.
# Then answer the following questions to be displayed on the console: [10 pts]
# The diamond with _____ cut has the maximum sales per count.
# The diamond with _____ cut has the minimum sales per count.

#%%
# GENERATING VARIABLES FOR EACH CUT TYPE
ideal_cut_count = len(diamond[diamond['cut'] == 'Ideal'])
premium_cut_count = len(diamond[diamond['cut'] == 'Premium'])
good_cut_count = len(diamond[diamond['cut'] == 'Good'])
verygood_cut_count = len(diamond[diamond['cut'] == 'Very Good'])
fair_cut_count = len(diamond[diamond['cut'] == 'Fair'])

print(ideal_cut_count)
print(premium_cut_count)
print(good_cut_count)
print(verygood_cut_count)
print(fair_cut_count)

#%%
cut_labels = diamond['cut'].unique()
cut_sales = [ideal_cut_count, premium_cut_count, verygood_cut_count,
             good_cut_count, fair_cut_count]

plt.figure(figsize=(18,10))
plt.barh(cut_labels, cut_sales, color='b', lw=2)
plt.title('SALES COUNT PER CUT', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND CUTS', fontsize=18)
plt.xticks(list(range(0,30000,2500)))
#plt.yticks()
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
print(f'DIAMONDS OF IDEAL CUT HAS THE MAXIMUM SALES PER COUNT ({ideal_cut_count} SALES).')
print(f'DIAMONDS OF FAIR CUT HAS THE MINIMUM SALES PER COUNT ({fair_cut_count} SALES).')

#%% [markdown]
# 6. Graph a horizontal bar-type plot that shows the diamond ‘color’ versus the sales .
# The y-axis should display various diamond ‘color’ and the x-axis should display the sales count per category.
# Add the title under ‘Sales count per color’.
# Then answer the following questions to be displayed on the console: [10pts]
# The diamond with _____ color has the maximum sales per count.
# The diamond with _____ color has the minimum sales per count.

#%%
# GENERATING VARIABLES FOR EACH CLARITY TYPE
# ['E', 'I', 'J', 'H', 'F', 'G', 'D']
E_color_count = len(diamond[diamond['color'] == 'E'])
I_color_count = len(diamond[diamond['color'] == 'I'])
J_color_count = len(diamond[diamond['color'] == 'J'])
H_color_count = len(diamond[diamond['color'] == 'H'])
F_color_count = len(diamond[diamond['color'] == 'F'])
G_color_count = len(diamond[diamond['color'] == 'G'])
D_color_count = len(diamond[diamond['color'] == 'D'])

print(E_color_count)
print(I_color_count)
print(J_color_count)
print(H_color_count)
print(F_color_count)
print(G_color_count)
print(D_color_count)

#%%
color_labels = diamond['color'].unique()
# REORDER?? REVERSE
color_sales = [E_color_count, I_color_count, J_color_count, H_color_count,
              F_color_count, G_color_count, D_color_count]

plt.figure(figsize=(18,10))
plt.barh(color_labels, color_sales, color='g', lw=2)
plt.title('SALES COUNT PER COLOR', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND COLOR', fontsize=18)
plt.xticks(list(range(0,13000,1000)))
#plt.yticks()
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
print(f'DIAMONDS OF G COLOR HAS THE MAXIMUM SALES PER COUNT ({G_color_count} SALES).')
print(f'DIAMONDS OF J COLOR HAS THE MINIMUM SALES PER COUNT ({J_color_count} SALES).')

#%% [markdown]
# 7. Graph a horizontal bar-type plot that shows the diamond ‘clarity versus the sales .
# The y-axis should display various diamond ‘clarity’ and the x-axis should display the sales count per category.
# Add the title under ‘Sales count per clarity’. Then answer the following questions to be displayed on the console: [10pts]
# The diamond with _____ clarity has the maximum sales per count.
# The diamond with _____ clarity has the minimum sales per count.

#%%
# GENERATING VARIABLES FOR EACH CLARITY TYPE
#['SI2', 'SI1', 'VS1', 'VS2', 'VVS2', 'VVS1', 'I1', 'IF']
SI2_clar_count = len(diamond[diamond['clarity'] == 'SI2'])
SI1_clar_count = len(diamond[diamond['clarity'] == 'SI1'])
VS1_clar_count = len(diamond[diamond['clarity'] == 'VS1'])
VS2_clar_count = len(diamond[diamond['clarity'] == 'VS2'])
VVS2_clar_count = len(diamond[diamond['clarity'] == 'VVS2'])
VVS1_clar_count = len(diamond[diamond['clarity'] == 'VVS1'])
I1_clar_count = len(diamond[diamond['clarity'] == 'I1'])
IF_clar_count = len(diamond[diamond['clarity'] == 'IF'])

print(SI2_clar_count)
print(SI1_clar_count)
print(VS1_clar_count)
print(VS2_clar_count)
print(VVS2_clar_count)
print(VVS1_clar_count)
print(I1_clar_count)
print(IF_clar_count)

#%%
clar_labels = diamond['clarity'].unique()
clar_sales = [SI2_clar_count, SI1_clar_count, VS1_clar_count, VS2_clar_count,
         VVS2_clar_count, VVS1_clar_count, I1_clar_count, IF_clar_count]
#explode = [.03, .03, .3, .03, .03]

plt.figure(figsize=(18,10))
plt.barh(clar_labels, clar_sales, color='k', lw=2)
plt.title('SALES COUNT PER CLARITY', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND CLARITY', fontsize=18)
plt.xticks(list(range(0,15000,1000)))
#plt.yticks()
plt.legend(loc='best')
plt.grid()
plt.show()

#%%
print(f'DIAMONDS OF SI1 CLARITY HAS THE MAXIMUM SALES PER COUNT ({SI1_clar_count} SALES).')
print(f'DIAMONDS OF I1 CLARITY HAS THE MINIMUM SALES PER COUNT ({I1_clar_count} SALES).')

#%%
# 8. Using the subplot command, plot the graph in the last 3 questions in one graph
# not shared axis, 1 row and 3 columns.
# Figure size: 16,8.  Each figure should have the title as defined above.  [10pts]

plt.figure(figsize=(16,8))
plt.subplot(1, 3, 1)
plt.barh(cut_labels, cut_sales, color='b', lw=2)
plt.title('SALES COUNT PER CUT', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND CUTS', fontsize=18)

plt.subplot(1, 3, 2)
plt.barh(color_labels, color_sales, color='g', lw=2)
plt.title('SALES COUNT PER COLOR', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND COLOR', fontsize=18)

plt.subplot(1, 3, 3)
plt.barh(clar_labels, clar_sales, color='k', lw=2)
plt.title('SALES COUNT PER CLARITY', fontsize=21)
plt.xlabel('SALES COUNT', fontsize=18)
plt.ylabel('DIAMOND CLARITY', fontsize=18)

plt.tight_layout(pad=1)
plt.show()

#%%
# 9. Plot the pie chart that displays the number of diamond sales per the cut.
# Explode value = 0.03.
# Add the title under ‘Sales count per cut in %’.
# Use the ‘equal’ and ‘square’ axis to plot the pie char in full circle.
# Display the percentage number on the pie chart .
# Then answer the following questions to be displayed on the console: [10pts]
# The diamond with _____ cut has the maximum sales per count with _____ % sales count.
# The diamond with _____ cut has the minimum sales per count with _____ % sales count.

ideal_cut_pct = (ideal_cut_count / len(diamond)) * 100
fair_cut_pct = (fair_cut_count / len(diamond)) * 100

explode_cut = [.03, .03, .03, .03, .03]

fig, ax = plt.subplots(1,1)
ax.pie(cut_sales, labels=cut_labels, explode=explode_cut, autopct='%1.2f%%')
plt.title('SALES COUNT PER CUT IN %', fontsize=21)
ax.axis('square')
ax.axis('equal')
plt.show()

#%%
print(f'DIAMONDS OF IDEAL CUT HAS THE MAXIMUM SALES PER COUNT WITH {ideal_cut_pct:.2f}% OF SALES.')
print(f'DIAMONDS OF FAIR CUT HAS THE MINIMUM SALES PER COUNT WITH {fair_cut_pct:.2f}% OF SALES.')

#%%
# 10.Plot the pie chart that displays the number of diamond sales per the color.
# Explode value = 0.03. Add the title under ‘Sales count per color’.
# Use the ‘equal’ and ‘square’ axis to plot the pie chart in full circle.
# Display the percentage number on the pie chart.
# Then answer the following questions to be displayed on the console: [10pts]
# The diamond with _____ color has the maximum sales per count with _____ % sales count.
# The diamond with _____ color has the minimum sales per count with _____ % sales count.

G_color_pct = (G_color_count / len(diamond)) * 100
J_color_pct = (J_color_count / len(diamond)) * 100

explode_color = [.03, .03, .03, .03, .03, .03, .03]

fig, ax = plt.subplots(1,1)
ax.pie(color_sales, labels=color_labels, explode=explode_color, autopct='%1.2f%%')
plt.title('SALES COUNT PER COLOR IN %', fontsize=21)
ax.axis('square')
ax.axis('equal')
plt.show()

#%%
print(f'DIAMONDS OF G COLOR HAS THE MAXIMUM SALES PER COUNT WITH {G_color_pct:.2f}% OF SALES.')
print(f'DIAMONDS OF J COLOR HAS THE MINIMUM SALES PER COUNT WITH {J_color_pct:.2f}% OF SALES.')

#%%
# 11.	Plot the pie chart that displays the number of diamond sales per the clarity.
# Explode value = 0.03. Add the title under ‘Sales count per clarity’.
# Use the ‘equal’ and ‘square’ axis to plot the pie char in full circle. Display the percentage number on the pie chart. Then answer the following questions to be displayed on the console: [10pts]
# The diamond with _____ clarity has the maximum sales per count with _____ % sales count.
# The diamond with _____ clarity has the minimum sales per count with _____ % sales count

SI1_clar_pct = (SI1_clar_count / len(diamond)) * 100
I1_clar_pct = (I1_clar_count / len(diamond)) * 100

explode_clar = [.03, .03, .03, .03, .03, .03, .03, .03]

fig, ax = plt.subplots(1,1)
ax.pie(clar_sales, labels=clar_labels, explode=explode_clar, autopct='%1.2f%%')
plt.title('SALES COUNT PER CLARITY IN %', fontsize=21)
ax.axis('square')
ax.axis('equal')
plt.show()

#%%
print(f'DIAMONDS OF SI1 CLARITY HAS THE MAXIMUM SALES PER COUNT WITH {SI1_clar_pct:.2f}% OF SALES.')
print(f'DIAMONDS OF I1 CLARITY HAS THE MINIMUM SALES PER COUNT WITH {I1_clar_pct:.2f}% OF SALES.')

#%%
# 12.	Using the subplot command, plot the graph in the last 3 questions in one graph ( not shared axis, 1 row and 3 columns).
# Figure size: 16,8. Each figure should have the title defined above.  [10pts]

plt.figure(figsize=(16,8))

plt.subplot(1, 3, 1)
fig, ax = plt.subplots(1,1)
ax.pie(cut_sales, labels=cut_labels, explode=explode_cut, autopct='%1.2f%%')
plt.title('SALES COUNT PER CUT IN %', fontsize=21)
#ax.axis('square')
#ax.axis('equal')



plt.subplot(1, 3, 2)
fig, ax = plt.subplots(1,1)
ax.pie(color_sales, labels=color_labels, explode=explode_color, autopct='%1.2f%%')
plt.title('SALES COUNT PER COLOR IN %', fontsize=21)
#ax.axis('square')
#ax.axis('equal')


plt.subplot(1, 3, 3)
fig, ax = plt.subplots(1,1)
ax.pie(clar_sales, labels=clar_labels, explode=explode_clar, autopct='%1.2f%%')
plt.title('SALES COUNT PER CLARITY IN %', fontsize=21)
ax.axis('square')
#ax.axis('equal')

#plt.tight_layout(pad=1)
plt.show()

#%%
## BONUS ATTEMPT
bonus = diamond.groupby(['cut', 'color']).mean()
print(bonus)