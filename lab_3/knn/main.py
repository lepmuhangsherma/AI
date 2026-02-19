from collections import Counter

import numpy as np


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        # kNN doesn't "train," it just memorizes the data
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 1. Compute distances between x and all examples in the training set
        distances = [np.sqrt(np.sum((x - x_train) ** 2)) for x_train in self.X_train]

        # 2. Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[: self.k]

        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Majority vote, most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# --- Example Usage ---
if __name__ == "__main__":
    # Sample Data: [Height, Weight]
    X_train = np.array([[170, 70], [180, 80], [150, 50], [160, 60]])
    y_train = np.array(["Large", "Large", "Small", "Small"])

    clf = KNN(k=3)
    clf.fit(X_train, y_train)

    # Predicting a new person [165, 65]
    new_data = np.array([[165, 65]])
    prediction = clf.predict(new_data)
    print(f"Prediction for [165, 65]: {prediction[0]}")
