import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# =========================================
# 1. LOAD DATA
# =========================================

data = load_iris()
X = data.data
y = data.target

# =========================================
# 2. PCA (DIMENSION REDUCTION)
# =========================================

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

print("Explained variance:", pca.explained_variance_ratio_)

plt.scatter(X_pca[:,0], X_pca[:,1], c=y)
plt.title("PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# =========================================
# 3. K-MEANS (SKLEARN)
# =========================================

kmeans_model = KMeans(n_clusters=3)
labels_kmeans = kmeans_model.fit_predict(X)

plt.scatter(X[:,0], X[:,1], c=labels_kmeans)
plt.title("K-Means (sklearn)")
plt.show()

# K-Means on PCA data (important addition)
labels_pca = kmeans_model.fit_predict(X_pca)

plt.scatter(X_pca[:,0], X_pca[:,1], c=labels_pca)
plt.title("K-Means on PCA Data")
plt.show()

# =========================================
# 4. K-MEANS FROM SCRATCH (NO CLASS)
# =========================================

def kmeans(X, k=3, max_iters=100):
    n_samples = len(X)

    # Random centroids
    idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[idx]

    for _ in range(max_iters):

        # Assign clusters
        clusters = []
        for x in X:
            distances = [np.linalg.norm(x - c) for c in centroids]
            clusters.append(np.argmin(distances))

        clusters = np.array(clusters)

        # Update centroids
        new_centroids = []
        for i in range(k):
            points = X[clusters == i]
            if len(points) > 0:
                new_centroids.append(np.mean(points, axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        # Stop if no change
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return clusters, centroids

# Run scratch K-Means
labels, centroids = kmeans(X, k=3)

plt.scatter(X[:,0], X[:,1], c=labels)
plt.scatter(centroids[:,0], centroids[:,1], marker='x', s=200)
plt.title("K-Means (Scratch)")
plt.show()
