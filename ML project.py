#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 13 20:53:24 2020

@author: bhaskaryuvaraj
"""
#project 1 machine learning
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import numpy as np
by=pd.read_csv('/Users/bhaskaryuvaraj/Downloads/carMPG.csv')

len(by)
len(by.columns)
#no. of rows=398, columns=9
by.head()
by.tail()
by.describe()
by.columns
by.dtypes
#horsepower is an numeric variable stored as object. hence has to be converted to numeric
by['Horsepower']=pd.to_numeric(by['Horsepower'],errors='coerce')
#by=by.replace({'?':np.nan}).dropna()  #alternate reference
by['Horsepower'].unique()


#by=by.dropna(subset=['Horsepower'])
by['Origin']=by['Origin'].replace([1,2,3],['USA make','German make','Japan make'])
#-----------------------------EDA---------------------------------

by.groupby(['Model_year','Origin'])['MPG'].mean().plot(kind='bar')
by.groupby('Cylinders')['MPG'].mean().plot(kind='line')

by.plot(kind='scatter',x='Acceleration',y='MPG',color='blue')


by.plot(kind='scatter',x='Horsepower',y='MPG',color='blue')

by.plot(kind='scatter',x='Weight',y='MPG',color='blue')
#the average MPG is more when the weight is less and vice versa

by.plot(kind='scatter',x='Displacement',y='MPG',color='blue')

#if you notice from the above codes. it is clear that weight, displacement and horsepower are corelated and i
#if any one increases all the other parameter shd increase and MPG dercreases accordingly.

#_____________________________EDA___________________________________ends

by.isnull().sum()
by.dropna(inplace=True)
#no missing values

#to check outliers

plt.boxplot(by['Weight'])
plt.boxplot(by['Acceleration']) #hasoutlier
plt.boxplot(by['Displacement'])
plt.boxplot(by['Horsepower'])

#outlier treatement
def remove_outlier(d,c):
    #find Q1
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    #find interquartile range
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    #filter data btw lb and ub
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

by=remove_outlier(by,'Acceleration')
plt.boxplot(by['Acceleration'])
by=remove_outlier(by,'Horsepower')
plt.boxplot(by['Horsepower']) #has an outlier again

#create dummy columns
dummy1=pd.get_dummies(by['Origin'])
dummy2=pd.get_dummies(by['Car_Name']) 
#concate both DF with by
ab=pd.concat([by,dummy1,dummy2], axis=1)

by1=ab.drop(ab.columns[[7,8]],axis=1)


#seperate the data
y=by1['MPG'].copy()
x=by1.drop(by1.columns[0],axis=1)

#create training and test data by splitting x and y into 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#fit the model
lm=linear_model.LinearRegression()
#creating the model object

#this is done using 70% training data sample
#create a training model
model=lm.fit(x_train,y_train)

#check the accuracy of training model
print(model.score(x_train,y_train))

#test the model using test data
#predict the price using x test
pred_y=lm.predict(x_test)

#check the prediction accuracy
print(model.score(x_test,y_test))