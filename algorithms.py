import numpy as np
import time
from collections import Counter
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix

# =====================================================
# Own implementation of KMeans
# =====================================================
class KMeans:
    def __init__(self, k, distance_metric='euclidean', max_iter=100):
        self.k = k # Number of clusters 
        self.distance_metric = distance_metric # [cite: 22, 23]
        self.max_iter = max_iter # Maximum number of iterations
        self.centroids = None

    def fit(self, data):
        # 1. Start with k random pixels as centers
        n_samples = data.shape[0]
        indices = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = data[indices]

        for i in range(self.max_iter):
            # 2. Group pixels by the closest center
            labels = self._get_labels(data)
            
            # 3. Move centers to the average of their group
            new_centroids = []

            for j in range(self.k):
                # Get all points that belong to the current cluster 'j'
                cluster_points = data[labels == j]
                
                # Check if the cluster actually has any points in it
                if len(cluster_points) > 0:
                    # Calculate the average (mean) of all points in this cluster
                    cluster_average = cluster_points.mean(axis=0)
                    new_centroids.append(cluster_average)
                else:
                    # If the cluster is empty, keep the old center so we don't crash
                    new_centroids.append(self.centroids[j])

            # Convert the list back into a solid NumPy array
            new_centroids = np.array(new_centroids)

            # 4. Stop if centers stop moving
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
        
        return labels

    def _get_labels(self, data):
        # Fast way to calculate distances for all pixels at once
        diff = data[:, np.newaxis, :] - self.centroids
        if self.distance_metric == 'euclidean':
            dist = np.sqrt(np.sum(diff**2, axis=2))
        elif self.distance_metric == 'manhattan':
            dist = np.sum(np.abs(diff), axis=2)
        elif self.distance_metric == 'maximum':
            dist = np.max(np.abs(diff), axis=2)
        return np.argmin(dist, axis=1)
    
# =====================================================
# Own implementation of DBSCAN
# =====================================================
class DBSCAN:
    def __init__(self, eps, min_samples, distance_metric='euclidean'):
        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric

    def fit(self, data):
        n_samples = data.shape[0]
        labels = np.full(n_samples, -1) 
        cluster_id = 0

        # Pre-calculate a distance matrix to speed things up (Universal N-dim)
        # we will find neighbors point-by-point
        for i in range(n_samples):
            if labels[i] != -1: 
                continue
            
            neighbors = self._get_neighbors(data, i)
            
            if len(neighbors) < self.min_samples:
                labels[i] = -1 
            else:
                self._expand_cluster(data, labels, i, neighbors, cluster_id)
                cluster_id += 1
        return labels

    def _get_neighbors(self, data, idx):
        diff = data - data[idx] 
        
        if self.distance_metric == 'euclidean':
            dist = np.sqrt(np.sum(diff**2, axis=1))
        elif self.distance_metric == 'manhattan':
            dist = np.sum(np.abs(diff), axis=1)
        elif self.distance_metric == 'maximum':
            dist = np.max(np.abs(diff), axis=1)
        return np.where(dist <= self.eps)[0]

    def _expand_cluster(self, data, labels, point_idx, neighbors, cluster_id):
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbors):
            nb_idx = neighbors[i]
            if labels[nb_idx] == -1: # If previously noise or unvisited
                labels[nb_idx] = cluster_id
                new_nb = self._get_neighbors(data, nb_idx)
                if len(new_nb) >= self.min_samples:
                    # If neighbor is also a core point, add its neighbors to the list
                    neighbors = np.unique(np.concatenate([neighbors, new_nb]))
            i += 1
            
            
# =====================================================
# Own implementation of KNN
# =====================================================
class KNNClassifier:
    def __init__(self, k=5):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [
                self._euclidean_distance(x, x_train)
                for x_train in self.X_train
            ]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = [self.y_train[i] for i in k_indices]
            predictions.append(
                Counter(k_labels).most_common(1)[0][0]
            )
        return np.array(predictions)


# =====================================================
# Visualization helper functions
# =====================================================

def plot_confusion_matrix(cm, class_names, title):
    plt.figure()
    plt.imshow(cm)
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(range(len(class_names)), class_names)
    plt.yticks(range(len(class_names)), class_names)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")

    plt.tight_layout()
    plt.show()


def plot_accuracy(results, dataset_name):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]

    plt.figure()
    plt.bar(models, accuracies)
    plt.title(f"Accuracy Comparison – {dataset_name}")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


def plot_runtime(results, dataset_name):
    models = list(results.keys())
    runtimes = [
        results[m]["train_time"] + results[m]["test_time"]
        for m in models
    ]

    plt.figure()
    plt.bar(models, runtimes)
    plt.title(f"Runtime Comparison – {dataset_name}")
    plt.ylabel("Time (seconds)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.show()


# =====================================================
# Experiment function
# =====================================================

def run_experiment(X, y, dataset_name, feature_names, class_names):
    print("\n" + "=" * 60)
    print(f"Dataset: {dataset_name}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    results = {}

    # ---------------- Own KNN ----------------
    knn_own = KNNClassifier(k=5)

    start = time.time()
    knn_own.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = knn_own.predict(X_test)
    test_time = time.time() - start

    results["Own KNN"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # ---------------- sklearn KNN ----------------
    knn_sklearn = KNeighborsClassifier(n_neighbors=5)

    start = time.time()
    knn_sklearn.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = knn_sklearn.predict(X_test)
    test_time = time.time() - start

    results["sklearn KNN"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # ---------------- Decision Tree ----------------
    tree = DecisionTreeClassifier(max_depth=3, random_state=42)

    start = time.time()
    tree.fit(X_train, y_train)
    train_time = time.time() - start

    start = time.time()
    y_pred = tree.predict(X_test)
    test_time = time.time() - start

    results["Decision Tree"] = {
        "accuracy": accuracy_score(y_test, y_pred),
        "train_time": train_time,
        "test_time": test_time,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }

    # ---------------- Print results ----------------
    for model, res in results.items():
        print(f"\nModel: {model}")
        print(f"Accuracy: {res['accuracy']:.4f}")
        print(f"Training time: {res['train_time']:.6f} s")
        print(f"Prediction time: {res['test_time']:.6f} s")
        print("Confusion Matrix:")
        print(res["confusion_matrix"])

    # ---------------- Confusion matrix plots ----------------
    for model, res in results.items():
        plot_confusion_matrix(
            res["confusion_matrix"],
            class_names,
            f"{model} – Confusion Matrix ({dataset_name})"
        )

    # ---------------- Comparison plots ----------------
    plot_accuracy(results, dataset_name)
    plot_runtime(results, dataset_name)

    # ---------------- Decision Tree visualization ----------------
    plt.figure(figsize=(16, 10))
    plot_tree(
        tree,
        feature_names=feature_names,
        class_names=class_names,
        filled=True,
        rounded=True
    )
    plt.title(f"Decision Tree – {dataset_name}")
    plt.show()

    def __init__(self, k, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        # KNN is a "lazy learner" - it just stores the data
        self.X_train = X
        self.y_train = y

    def predict(self, X_test):
        predictions = [self._predict_single(x) for x in X_test]
        return np.array(predictions)

    def _predict_single(self, x):
        # 1. Compute distances between x and all examples in the training set
        if self.distance_metric == 'euclidean':
            distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        elif self.distance_metric == 'maximum':
            distances = np.max(np.abs(self.X_train - x), axis=1)

        # 2. Sort distances and return indices of the k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]

        # 3. Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # 4. Majority vote: find the most common label
        # (Using a simple count logic for beginners)
        counts = {}
        for label in k_nearest_labels:
            counts[label] = counts.get(label, 0) + 1
        
        # Return the label with the highest count
        return max(counts, key=counts.get)