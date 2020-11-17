# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 19:33:32 2020

@author: Shashwat
"""
# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
from scipy.linalg import eigh
from sklearn import decomposition

# loading the data
digits = datasets.load_digits()

# converting the loaded data into dataframe
data = pd.DataFrame(digits.data, columns=digits.feature_names)
data['label'] = pd.Series(digits.target)

# print top 10 rows of data
print(data.head())

# removing label from dataset and storing it into diffrent list
label = data['label']
data = data.drop('label', axis=1)

# print shape of data
print(data.shape)       #(1797, 64)
print(label.shape)      #(1797, )


# to plot an image and print its value
plt.figure(figsize=(8,8))
idx = 3
grid_data = data.iloc[idx].values.reshape(8,8)  
plt.imshow(grid_data, interpolation = "none", cmap = "gray")
plt.show()
print(label[idx])

# standarize the data
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)      #(1797, 64)

# find covariance matrix
sample_data = standardized_data
covar_matrix = np.matmul(sample_data.T , sample_data)
print("The shape of co-variance matrix = ", covar_matrix.shape)       #(64, 64)

# Find Eigen values and Eigen Vectors for Co-variance matrix
values, vectors = eigh(covar_matrix, eigvals=(62,63))
print("Shape of eigen vectors = ",vectors.shape)   #(64, 2)
vectors = vectors.T
print("Updated shape of eigen vectors = ",vectors.shape)    #(2, 64)
# here the vectors[1] represent the eigen vector corresponding 1st principal eigen vector
# here the vectors[0] represent the eigen vector corresponding 2nd principal eigen vector

# Project original data points on to the new plane
new_coordinates = np.matmul(vectors, sample_data.T)
print("resultanat new data points' shape ", vectors.shape, "X", sample_data.T.shape," = ", new_coordinates.shape)

# Add the label column to the new data
new_coordinates = np.vstack((new_coordinates, label)).T
dataframe = pd.DataFrame(data=new_coordinates, columns=("1st_principal", "2nd_principal", "label"))

# Visualizing data using the new 2-D features
sns.FacetGrid(dataframe, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()

# PCA using scikit learn
pca = decomposition.PCA()
pca.n_components = 2
pca_data = pca.fit_transform(sample_data)
pca_data = np.vstack((pca_data.T, label)).T
pca_df = pd.DataFrame(data=pca_data, columns=("1st_principal", "2nd_principal", "label"))
sns.FacetGrid(pca_df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
plt.show()