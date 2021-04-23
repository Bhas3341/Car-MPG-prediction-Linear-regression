# -*- coding: utf-8 -*-
"""
Created on Fri Jul  3 13:12:31 2020

@author: Bhaskar
"""

#importing the file
import pandas as pd
df=pd.read_excel("C:\\Users\\Bhaskar\\Desktop\\carMPG 2.xlsx")
print(df)

#check the number of rows n column
df.shape

#check the datatypes
df.dtypes

#Do any conversion required
df['Horsepower']=pd.to_numeric(df['Horsepower'],errors='coerce')
df.dtypes

#check the missing value
df.isnull().sum()

#do the the missing treatment if reqd
df=df.dropna()

#prepare summary report
df.describe()

#understand the data by graphs
#check the outlier

#do the the treatment if reqd
#Future engineering
#conversion of all char to numeric i.e dummy
#split the data into train n test
#use suitable ML Model
#find the accuracy


import seaborn as sns
sns.distplot(df['Horsepower'],bins=15,hist=False,kde=True,color='Green')
plt.xlabel('Horsepower',fontsize=12)
plt.ylabel('Frequency',fontsize=15)
plt.title(('Price Distribution curve'))
plt.show
import matplotlib.pyplot as plt
plt.boxplot(df['Horsepower'])
q1=df['Horsepower'].quantile(0.25)
q3=df['Horsepower'].quantile(0.75)
iqr=q3-q1
ub=q3+(1.5*iqr)
df['Horsepower'][df['Horsepower']>ub]=ub

import seaborn as sns
sns.distplot(df['Displacement'],bins=15,hist=False,kde=True,color='Green')
plt.xlabel('Displacement',fontsize=12)
plt.ylabel('Frequency',fontsize=15)
plt.title(('Price Distribution curve'))
plt.show
import matplotlib.pyplot as plt
plt.boxplot(df['Displacement'])
q1=df['Displacement'].quantile(0.25)
q3=df['Displacement'].quantile(0.75)
iqr=q3-q1
ub=q3+(1.5*iqr)
df['Displacement'][df['Displacement']>ub]=ub

import seaborn as sns
sns.distplot(df['Cylinders'],bins=15,hist=False,kde=True,color='Green')
plt.xlabel('Cylinders',fontsize=12)
plt.ylabel('Frequency',fontsize=15)
plt.title(('Price Distribution curve'))
plt.show
import matplotlib.pyplot as plt
plt.boxplot(df['Displacement'])
q1=df['Displacement'].quantile(0.25)
q3=df['Displacement'].quantile(0.75)
iqr=q3-q1
ub=q3+(1.5*iqr)
df['Displacement'][df['Displacement']>ub]=ub

import seaborn as sns
sns.distplot(df['Weight'],bins=15,hist=False,kde=True,color='Green')
plt.xlabel('Weight',fontsize=12)
plt.ylabel('Frequency',fontsize=15)
plt.title(('Price Distribution curve'))
plt.show
import matplotlib.pyplot as plt
plt.boxplot(df['Weight'])
q1=df['Weight'].quantile(0.25)
q3=df['Weight'].quantile(0.75)
iqr=q3-q1
ub=q3+(1.5*iqr)
df['Weight'][df['Weight']>ub]=ub

