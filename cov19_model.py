#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 25 14:36:35 2020

@author: s.p.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dateutil import parser
from heatmap import corrplot, heatmap
%matplotlib qt

dataset = pd.read_csv('cov19_clean.csv')

#Visualize the data
dataset.info()
dataset.describe()
dataset.head()

#Correlation Matrix
dataset.iloc[:,:-1].corrwith(dataset.InfectRate).plot.line()
plt.xticks(ticks = range(len(dataset.columns)-1), labels = ['population_density', 'median_age', 'aged_65_older', 'aged_70_older',
       'gdp_per_capita', 'cvd_death_rate', 'diabetes_prevalence',
       'female_smokers', 'male_smokers', 'handwashing_facilities',
       'hospital_beds_per_100k'],rotation = 45)
plt.title('Variable correlating to the Infection Rate ')

corrplot(dataset.corr(), size_scale=500, marker='s')

#Data Preprocessing
column_names_x = pd.DataFrame(dataset.iloc[:, 1:-1].columns.values)
x = pd.DataFrame(dataset.iloc[:, 1:-1].values)
y = dataset.iloc[:, -1].values
x.columns = column_names_x.values
#Feature scaling is not necessary for this Linear Regression 

#Split into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = .2, random_state = 0)

#Apply Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the test results
y_pred = regressor.predict(x_test)
np.set_printoptions(precision = 2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


#Accuracy of the result
from sklearn.metrics import r2_score, mean_squared_error
from math import sqrt
rms = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test,y_pred)
print ('RMSE: %0.3f R2 : %0.3f' %(rms,r2))


#OLS visualization with backward propagation
import statsmodels.api as sm
x = sm.add_constant(x).astype(float)
x_opt = x.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,1,2,3,4,5,6,7,8,9,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,1,2,3,4,5,6,8,9,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,1,2,3,4,5,8,9,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,1,3,4,5,8,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,3,4,5,8,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,3,5,8,11]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()

x_opt = x.iloc[:, [0,3,5,8]]
regressor_ols = sm.OLS(y,x_opt).fit()
regressor_ols.summary()




