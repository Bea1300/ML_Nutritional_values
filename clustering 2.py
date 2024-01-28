import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
dataset_path = 'dataset/food.csv'  
df = pd.read_csv(dataset_path)

# Clean up column names
df.columns = df.columns.str.strip()

# Drop non-numeric columns if needed
df_numeric = df.select_dtypes(include='number')

# Standardize the data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# Choose the number of clusters (consider trying different values)
num_clusters = 2

# Initialise and fit the k-means model
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Visualise the clusters
for cluster in range(num_clusters):
    cluster_data = df[df['Cluster'] == cluster]
    plt.scatter(cluster_data['Data.Fiber'], cluster_data['Data.Kilocalories'], label=f'Cluster {cluster}')

# Plot centroids
for cluster in range(num_clusters):
    centroid_x = cluster_data['Data.Fiber'].mean()
    centroid_y = cluster_data['Data.Kilocalories'].mean()
    plt.scatter(centroid_x, centroid_y, s=50, c='red', marker='X', label=f'Centroid {cluster}')

plt.xlabel('Fiber')
plt.ylabel('Kilocalories')
plt.title('K-Means Clustering')
plt.legend()
plt.show()
