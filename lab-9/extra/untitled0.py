# -*- coding: utf-8 -*-
"""
Created on Thu Oct  1 09:52:18 2020

@author: Shashwat
"""

# importing librarires
import pandas as pd
import numpy as np        
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn import metrics

# loading the data
cancer = datasets.load_breast_cancer()
df = pd.DataFrame(cancer.data, columns=cancer.feature_names)
df['target'] = pd.Series(cancer.target)

# defining dependent and independent variable
X_Col = list(df.columns)
X_Col.remove('target')
Y_Col = 'target'

# train test split of dataset
X_train, X_test, y_train, y_test = train_test_split(df[X_Col], df[Y_Col], test_size=0.3)

# creating SVM model and training it on our data
SVM = SVC(kernel='linear', random_state=0, gamma=.10, C=1.0)
SVM.fit(X_train, y_train)

# predicting
y_pred = SVM.predict(X_test)

# Model information
print("Accuracy of training data ; ", SVM.score(X_train, y_train))
print("Accuracy of testing data ; ", SVM.score(X_test, y_test))

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))