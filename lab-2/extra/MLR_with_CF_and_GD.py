# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 09:31:57 2020

@author: Shashwat
"""


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

'''
CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
RM - average number of rooms per dwelling
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's
'''

# X = RM + LSTAT + PTRATIO

#read data
data = pd.read_csv("Boston.csv")

#Print head and description of data
'''
print(data.head())
print(data.describe())
'''

#Storing dependent and independent variable in seprate numpy array
x = np.array((data["rm"], data["lstat"], data["ptratio"]))
x = x.transpose()
#X is 2D array of shape 506x3, in which each column represent one variable

y = np.array(data["medv"])
y = y.reshape(y.shape[0],1)


m = len(x)   #length of data
theta0 = 0   #slope of rm
theta1 = 0   #slope of lstat
theta2 = 0   #slope of ptratio
intercept = 22.5012   #intercept

coeff = np.array((theta0, theta1, theta2))
coeff = coeff.reshape(coeff.shape[0],1)


def cost_func(C, I, X, Y):
    #C -> its a coeff. matrix of shape 3x1 in this case
    #I -> its an intercept value
    #X -> its a data input matrox of shape 506x3 in this case
    #Y -> its a target variable matrix of shape 506x1 in this case
    cost = (1/(2*len(X))) *  np.sum(np.square(np.matmul(X, C) + I - Y))
    return cost    


iteration = 20000
alpha = 0.000001
#if we take any value of alpha bigger than this, cost increases drastically

for i in range(iteration):
    y_pred = np.matmul(x, coeff) + intercept
    
    d_theta0 =  (-2/m)*np.sum(x[:,0]*(y - y_pred))
    d_theta1 =  (-2/m)*np.sum(x[:,1]*(y - y_pred))
    d_theta2 =  (-2/m)*np.sum(x[:,2]*(y - y_pred))
    
    d_intercept = (-2/m)*np.sum(y - y_pred)
    
    coeff[0] = coeff[0] - alpha*d_theta0
    coeff[1] = coeff[1] - alpha*d_theta1
    coeff[2] = coeff[2] - alpha*d_theta2
    intercept = intercept - alpha*d_intercept
    cost = cost_func(coeff, intercept, x, y)
    if(i%1000==0):
        print(i,"---",coeff,"---",intercept,"---",cost)
        
        

'''        
#Same model using sklearn
X = data[['rm', 'lstat', 'ptratio']]
Y = data['medv']
model = LinearRegression()  
model.fit(X,Y)
print(model.intercept_)
print(model.coef_)

# Coeff -> [ 4.51542094 -0.57180569 -0.93072256]
# Intercept -> 18.567111505395264 
# using these coeff and intercept values, we get cost fucntion value as 13.56520287924854

abc = np.array((4.51542094, -0.57180569, -0.93072256))
abc = abc.reshape(abc.shape[0],1)
bcd = 18.567111505395264
print(cost_func(abc, bcd, x, y))
'''

'''
# 22.5 -> 42.21
abc = np.array((0, 0, 0))
abc = abc.reshape(abc.shape[0],1)
bcd = 22.40
print(cost_func(abc, bcd, x, y))
'''    
