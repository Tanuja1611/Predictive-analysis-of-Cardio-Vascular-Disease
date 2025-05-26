#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install necessary libraries
# In your command prompt, run:
# pip install pandas numpy scikit-learn matplotlib scikit-learn-extra

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Step 1: Load your dataset
file_path = "C:\\Users\\anush\\Desktop\\CLASS NOTES\\minor\\CVD.csv"  # Replace with the correct path to your CSV file
data = pd.read_csv(file_path)

# Step 2: Preprocess - Drop any non-numerical or irrelevant columns if necessary
# Assuming all columns are relevant for clustering, if you need to drop specific columns, do it here.
# Example: Drop non-numeric columns
data_numeric = data.select_dtypes(include=[np.number])  # Select only numeric columns

# Step 3: Standardize the dataset
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_numeric)

# Step 4: Apply K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_scaled)

# Step 5: Apply K-Medoids Clustering
kmedoids_model = KMedoids(n_clusters=3, random_state=42)
y_kmedoids = kmedoids_model.fit_predict(X_scaled)

# Step 6: Visualize K-Means Clusters
plt.figure(figsize=(10, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmeans, s=50, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X')
plt.title("K-Means Clustering")
plt.show()

# Step 7: Visualize K-Medoids Clusters
plt.figure(figsize=(10, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_kmedoids, s=50, cmap='viridis')
plt.scatter(X_scaled[kmedoids_model.medoid_indices_, 0], X_scaled[kmedoids_model.medoid_indices_, 1], s=200, c='blue', marker='X')
plt.title("K-Medoids Clustering")
plt.show()

# Step 8: Evaluate Clustering Performance using Silhouette Score
silhouette_kmeans = silhouette_score(X_scaled, y_kmeans)
silhouette_kmedoids = silhouette_score(X_scaled, y_kmedoids)

print(f"Silhouette Score for K-Means: {silhouette_kmeans}")
print(f"Silhouette Score for K-Medoids: {silhouette_kmedoids}")


# In[ ]:





# In[ ]:




