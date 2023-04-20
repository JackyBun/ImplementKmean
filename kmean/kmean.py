import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = X[np.random.choice(range(len(X)), self.n_clusters)]
        
        for i in range(self.max_iter):
            self.clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                self.clusters[np.argmin(distances)].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(c, axis=0) for c in self.clusters]

            if np.allclose(self.centroids, prev_centroids):
                break

    def predict(self, X):
        distances = [np.linalg.norm(X - c, axis=1) for c in self.centroids]
        return np.argmin(distances, axis=0)

