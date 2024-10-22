import numpy as np
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Generate random data with two classes
X, Y = make_blobs(n_samples=16, n_features=2, centers=2, center_box=(-2, 2))
# Scale the data so that all values are between 0.0 and 1.0
X = MinMaxScaler().fit_transform(X)
# Split two data points from the data as test data and use the remaining n-2 points as the training data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=2)

# Place-holder for the predicted classes
y_predict = np.empty(len(y_test), dtype=np.int64)

# Produce line segments that connect the test data points to the nearest neighbors for drawing the chart
lines = []

# Distance function
def dist(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def main(X_train, X_test, y_train, y_test):
    global y_predict
    global lines
    k = 3  # Number of nearest neighbors

    for i, test_item in enumerate(X_test):
        # Calculate the distances to all training points
        distances = [dist(train_item, test_item) for train_item in X_train]

        # Get the indices of the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = y_train[nearest_indices]

        # Predict class based on the majority of nearest labels
        y_predict[i] = int(np.round(np.mean(nearest_labels)))  # Convert to int

        # Create lines connecting the test point to the nearest neighbors (for visualization)
        for index in nearest_indices:
            lines.append(np.stack((test_item, X_train[index])))

    print(y_predict)

main(X_train, X_test, y_train, y_test)
