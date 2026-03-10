#Mall Customer Dataset

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os

file_path = r"C:\Users\Admin\Desktop\Python\Mall Customer Dataset\Mall_Customers.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Oops! File not found at: {file_path}")

data = pd.read_csv(file_path)

print("Dataset has", data.shape[0], "rows and", data.shape[1], "columns")
print("First 5 rows of the dataset:")
print(data.head())

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

wcss = []  

max_clusters = min(10, len(X))

for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,5))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--', color='blue')
plt.title("Elbow Method to Find Best Number of Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
plt.grid(True)
plt.show()

optimal_clusters = 5
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
y_kmeans = kmeans.fit_predict(X)

plt.figure(figsize=(8,5))

plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, cmap='rainbow', s=50)

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='black', label='Centroids')

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments (K-Means Clustering)")
plt.legend()
plt.grid(True)
plt.show()