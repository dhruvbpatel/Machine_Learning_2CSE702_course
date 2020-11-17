# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:23:06 2020

@author: Shashwat
"""

# importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator

# loading the data
data = pd.read_csv("wine_data.csv")

# Visualization
plt.scatter(x = 'Alcohol', y = 'OD', c= 'Wine', data = data)
plt.xlabel("Alcohol")
plt.ylabel("OD")
plt.show()

# apply kmeans on the data
kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
kmeans = KMeans(n_clusters=3, **kmeans_kwargs).fit(data.iloc[:,[12,1]])
centroids_df = pd.DataFrame(kmeans.cluster_centers_, columns = list(data.iloc[:,[12,1]].columns.values))
fig, ax = plt.subplots(1, 1)
data.plot.scatter(x = 'Alcohol', y = 'OD', c= kmeans.labels_, figsize=(12,8), colormap='jet', ax=ax, mark_right=False)
centroids_df.plot.scatter(x = 'Alcohol', y = 'OD', ax = ax,  s = 80, mark_right=False)

 
# OPTIONAL
# to select the number of clusters
kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}
sse = []
for i in range(1,10,1):
    kmeans = KMeans(n_clusters=i, **kmeans_kwargs).fit(data.iloc[:,[12,1]])
    sse.append(kmeans.inertia_)
plt.style.use("fivethirtyeight")
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.plot(range(1,10,1), sse)
plt.show()
# to select number of cluster
kl = KneeLocator(range(1, 10, 1), sse, curve="convex", direction="decreasing")
print(kl.elbow)

    