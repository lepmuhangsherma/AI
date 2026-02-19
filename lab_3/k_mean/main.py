import numpy as np


class KMeans:
    def __init__(self, k=3, max_iters=100):
        self.k = k
        self.max_iters = max_iters

    def fit(self, X):
        # 1. Randomly initialize centroids from the data points
        self.centroids = X[np.random.choice(len(X), self.k, replace=False)]

        for _ in range(self.max_iters):
            # 2. Assign points to the nearest centroid
            # Calculating Euclidean distance between points and centroids
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)

            # 3. Calculate new centroids (the mean of the points in each cluster)
            new_centroids = np.array(
                [X[labels == i].mean(axis=0) for i in range(self.k)]
            )

            # 4. Check for convergence (if centroids didn't move, we're done)
            if np.all(self.centroids == new_centroids):
                break

            self.centroids = new_centroids

        return labels

X = np.array([[150, 50], [152, 53], [185, 90], [188, 95], [151, 51], [190, 92]])

model = KMeans(k=2)
cluster_assignments = model.fit(X)

print("Cluster Assignments:", cluster_assignments)

