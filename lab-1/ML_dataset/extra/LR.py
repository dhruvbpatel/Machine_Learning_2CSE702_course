# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 09:19:45 2020

@author: Shashwat
"""

#importing librarires
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics


#load the data file
data = pd.read_csv("IceCreamData.csv")

#Print head and description of data
'''
print(data.head())
print(data.describe())
'''

#Visualization of data 
'''
ax = sns.scatterplot(x="Temperature", y="Revenue", data=data)
'''

#DO NOT RUN THIS BLOCK
#Normalize the data and store it into dataframe
'''
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(data)
data = pd.DataFrame()
data['Temperature'] = scaled_data[:,0]
data['Revenue'] = scaled_data[:,1]
'''

#Split the data into training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data['Temperature'], data['Revenue'], test_size=0.2, random_state=0)

X_train = np.array(X_train).reshape(X_train.shape[0],1)
y_train = np.array(y_train).reshape(y_train.shape[0],1)

X_test = np.array(X_test).reshape(X_test.shape[0],1)
y_test = np.array(y_test).reshape(y_test.shape[0],1)


#train the model
model = LinearRegression()  
model.fit(X_train, y_train)

#Intercept and coefficent of variable in noramalized form
print(model.intercept_)
print(model.coef_)



#predict 
y_predict = model.predict(X_test) 

#store actual and predicted values in compare dataframe
compare = pd.DataFrame()
compare["actual value"] = y_test.flatten()
compare["predicted value"] = y_predict.flatten()

#Visulaization of actual and predicted values
#sns.relplot(data=compare);
#sns.pairplot(data, x_vars='Temperature', y_vars="Revenue",size=7, aspect=0.7, kind='reg')

#Error
print(metrics.mean_absolute_error(compare["actual value"], compare["predicted value"]))
print(metrics.mean_squared_error(compare["actual value"], compare["predicted value"]))
print(np.sqrt(metrics.mean_squared_error(compare["actual value"], compare["predicted value"])))

