import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering

# =========================================
# 1. LOAD DATA (IRIS)
# =========================================

data = load_iris()
X = data.data
y = data.target

# =========================================
# 2. PCA (DIMENSION REDUCTION)
# =========================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Plot PCA result
plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.title("PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# =========================================
# 3. K-MEANS (SKLEARN)
# =========================================

kmeans = KMeans(n_clusters=3)
labels_kmeans = kmeans.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels_kmeans)
plt.title("K-Means Clustering")
plt.show()

# =========================================
# 4. HIERARCHICAL CLUSTERING
# =========================================

hc = AgglomerativeClustering(n_clusters=3)
labels_hc = hc.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels_hc)
plt.title("Hierarchical Clustering")
plt.show()

# =========================================
# 5. K-MEANS FROM SCRATCH
# =========================================

class SimpleKMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        n_samples, n_features = X.shape

        # random centroids
        idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = X[idx]

        for _ in range(self.max_iters):
            # assign clusters
            clusters = []
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                cluster = np.argmin(distances)
                clusters.append(cluster)

            clusters = np.array(clusters)

            # update centroids
            new_centroids = []
            for i in range(self.k):
                points = X[clusters == i]
                if len(points) > 0:
                    new_centroids.append(np.mean(points, axis=0))
                else:
                    new_centroids.append(self.centroids[i])

            new_centroids = np.array(new_centroids)

            # stop if no change
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        self.labels_ = clusters

    def predict(self, X):
        labels = []
        for x in X:
            distances = [np.linalg.norm(x - c) for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

# Train scratch KMeans
kmeans_scratch = SimpleKMeans(k=3)
kmeans_scratch.fit(X)

plt.scatter(X[:,0], X[:,1], c=kmeans_scratch.labels_)
plt.title("K-Means (Scratch)")
plt.show()