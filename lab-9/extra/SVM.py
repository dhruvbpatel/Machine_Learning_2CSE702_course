#!/usr/bin/env python
# coding: utf-8

# In[22]:


## Lab -9 SVM  17162121014 Dhruv Patel
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


## import model related functions
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix,plot_confusion_matrix,recall_score,precision_score
from sklearn.preprocessing import StandardScaler,MinMaxScaler

##  tasks
from sklearn import datasets
cancer = datasets.load_breast_cancer()


# In[99]:


# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant' 'benign')
print("Labels: ", cancer.target_names)


# In[100]:


# print data(feature)shape
cancer.data.shape


# In[101]:


print(cancer.data[0:2])


# In[102]:


print(cancer.target)


# In[103]:


X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test


# In[104]:


svm = SVC(kernel="linear",random_state=0)


# In[105]:


svm.fit(X_train,y_train)


# In[106]:


y_pred = svm.predict(X_test)


# In[108]:


print("Accuracy:",accuracy_score(y_test, y_pred))


# In[111]:


# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",recall_score(y_test, y_pred))


# In[ ]:




