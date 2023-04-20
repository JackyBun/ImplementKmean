import numpy as np

class KMeansPP:
    def __init__(self, n_clusters=3, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.centroids = [X[np.random.choice(len(X))]]
        while len(self.centroids) < self.n_clusters:
            distances = np.array([min([np.linalg.norm(x - c)**2 for c in self.centroids]) for x in X])
            probs = distances / np.sum(distances)
            self.centroids.append(X[np.random.choice(len(X), p=probs)])

        for i in range(self.max_iter):
            self.clusters = [[] for _ in range(self.n_clusters)]
            for x in X:
                distances = [np.linalg.norm(x - c) for c in self.centroids]
                self.clusters[np.argmin(distances)].append(x)

            prev_centroids = self.centroids
            self.centroids = [np.mean(c, axis=0) for c in self.clusters]

            if np.allclose(self.centroids, prev_centroids):
                break

    def _calculate_distances(self, X):
      return np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)

    def predict(self, X):
      return np.argmin(self._calculate_distances(X), axis=1)
