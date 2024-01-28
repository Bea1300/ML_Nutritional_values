import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset into a Pandas DataFrame
dataset_path = 'dataset/food.csv'
df = pd.read_csv(dataset_path)

# Clean up column names
df.columns = df.columns.str.strip()

# Feature Selection
selected_features = ['Data.Fiber', 'Data.Kilocalories']
df_selected = df[selected_features]

# Data Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected)

# Elbow Method to find the optimal number of clusters
wcss = []  # Within-cluster sum of squares

# Experiment with a range of k values
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.show()

# Choose the optimal number of clusters based on the elbow method
optimal_num_clusters = 5  

# Initialise and fit the k-means model with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(df_scaled)

# Transform centroids back to the original scale
centroids_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)

# Visualise the clusters with properly scaled centroids
plt.scatter(df_selected['Data.Fiber'], df_selected['Data.Kilocalories'], c=df['Cluster'], cmap='viridis', s=10)
plt.scatter(centroids_original_scale[:, 0], centroids_original_scale[:, 1], s=300, c='red', marker='X', label='Centroids')
plt.xlabel('Fiber')
plt.ylabel('Kilocalories')
plt.title(f'K-Means Clustering with {optimal_num_clusters} Clusters (Elbow Method)')
plt.legend()
plt.show()
