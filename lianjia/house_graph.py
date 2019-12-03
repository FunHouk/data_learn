import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib as mpl 

import matplotlib.pyplot as plt 
from IPython.display import display
plt.style.use('fivethirtyeight')
sns.set_style('ticks',{'font.sans-serif':['simhei','Arial']})

lianjia_df = pd.read_csv('C:\data pra\lianjia.csv')
#display(lianjia_df.head(n=2))
#lianjia_df.info()
#lianjia_df.describe()

df = lianjia_df.copy()
df['PerPrice'] = lianjia_df['Price'] / lianjia_df['Size']

columns = ['Region', 'District', 'Garden', 'Layout', 'Floor', 'Year', 'Size', 'Elevator', 'Direction', 'Renovation', 'PerPrice', 'Price']
df = pd.DataFrame(df,columns= columns)
df = df [(df['Layout']!='叠拼别墅')&(df['Size']<1000)]

#display(df.head(n=2))

df_house_count = df.groupby('Region')['Price'].count().sort_values(ascending=False).to_frame().reset_index() #各区域价格个数，即各区域二手房套数
df_house_mean = df.groupby('Region')['PerPrice'].mean().sort_values(ascending=False).to_frame().reset_index() #各区域平均价格

'''
f, [ax1,ax2,ax3] = plt.subplots(3,1,figsize=(15,10))

sns.barplot(x='Region',y='PerPrice',palette='Blues_d',data=df_house_mean,ax=ax1)
ax1.set_title('北京各大区二手房每平米单价对比',fontsize=15)
ax1.set_xlabel('Region',fontsize=5)
ax1.set_ylabel('PerPrice')

sns.barplot(x='Region',y='Price',palette='Greens_d',data=df_house_count,ax=ax2)
ax2.set_title('北京各大区二手房数量对比',fontsize=15)
ax2.set_xlabel('Region',fontsize=5)
ax2.set_ylabel('Count')

sns.boxplot(x='Region',y='Price',data=df,ax=ax3)
ax3.set_title('北京各大区二手房房屋总价',fontsize=15)
ax3.set_xlabel('Region',fontsize=5)
ax3.set_ylabel('Price')

plt.tight_layout()
plt.show()
'''


f,[ax1,ax2] = plt.subplots(1,2,figsize=(15,5))

sns.distplot(df['Size'],bins=20,ax=ax1,color='r')
sns.kdeplot(df['Size'],shade=True,ax=ax1)

sns.regplot(x='Size',y='Price',data=df,ax=ax2)

plt.show()


'''
f,ax1 = plt.subplots(figsize=(20,20))
sns.countplot(y='Layout',data=df,ax=ax1)
ax1.set_title('房屋户型',fontsize=15)
ax1.set_xlabel('Count')
ax1.set_ylabel('Layout')
plt.show()
'''


f,[ax1,ax2,ax3] = plt.subplots(1,3,figsize=(20,5))
sns.countplot(df['Renovation'],ax=ax1)
sns.barplot(x='Renovation',y='Price',data=df,ax=ax2)
sns.boxplot(x='Renovation',y='Price',data=df,ax=ax3)
plt.show()


'''
misn = len(df.loc[(df['Elevator'].isnull()),'Elevator'])
print('Elevator缺失值数量为：'+ str(misn))

df['Elevator'] = df.loc[(df['Elevator'] == '有电梯')|(df['Elevator']=='无电梯'),'Elevator'] # loc(行的特征，列的特征)

df.loc[(df['Floor']>6)&(df['Elevator'].isnull()),'Elevator'] = '有电梯'
df.loc[(df['Floor']<=6)&(df['Elevator'].isnull()), 'Elevator'] = '无电梯'

f,[ax1,ax2] = plt.subplots(1,2 ,figsize=(20,10))
sns.countplot(df['Elevator'],ax=ax1)
ax1.set_title(' ',fontsize=15)
ax1.set_xlabel(' ')
ax1.set_ylabel(' ')
sns.barplot(x='Elevator',y='Price',data=df,ax=ax2)
ax2.set_title(' ',fontsize=15)
ax2.set_xlabel('')
ax2.set_ylabel('')
plt.show()
'''

'''
grid = sns.FacetGrid(df,row='Elevator', col='Renovation',palette='seismic',height=4)
grid.map(plt.scatter,'Year','Price')
grid.add_legend()
plt.tight_layout()
plt.show()
'''





