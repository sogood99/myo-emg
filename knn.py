from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch
import matplotlib.pyplot as plt
import sys

from data import Label


class Classifier:
    def __init__(self, X, y, model, k=3):
        self.k = k
        self.model = model
        self.model.eval()
        self.X = X
        self.y = y

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.X = self.model(self.X).detach().numpy()

        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X, self.y)

        distances = self.knn.kneighbors(self.X, n_neighbors=k, return_distance=True)

        mean_distance = np.mean(distances[0], axis=1)

        # self.threshold = np.percentile(mean_distance, 95) + 3 * np.std(mean_distance)
        self.threshold = 1000

    def predict(self, X):
        X = torch.tensor(X, dtype=torch.float32)
        X = self.model(X).detach().numpy()

        distances = self.knn.kneighbors(X, n_neighbors=self.k, return_distance=True)
        distances = np.mean(distances[0], axis=1)

        pred = self.knn.predict(X)
        pred[distances > self.threshold] = Label.Unknown

        return pred


if len(sys.argv) > 1 and sys.argv[1] == "plot":
    X = np.load("data/samples.npy")
    y = np.load("data/labels.npy").squeeze()
    model, _ = torch.load("model.pth")
    model = model[:3]

    cls = Classifier(X, y, model)

    X_unknown = np.load("data/unknown.npy")

    print(cls.predict(X_unknown))  # 19/20 unknown

    # X_unknown = torch.tensor(X_unknown, dtype=torch.float32)
    # X_unknown = model(X_unknown).detach().numpy()
    # plt.hist(mean_distance, bins=100, label="In-Distribution Data Distance")

    # plt.axvline(threshold, color="red", linestyle="--", label="Threshold")

    # distances_unknown = knn.kneighbors(X_unknown, n_neighbors=k, return_distance=True)
    # distances_unknown = np.mean(distances_unknown[0], axis=1)

    # plt.hist(distances_unknown, bins=150, label="OOD Data Distance")
    # plt.legend()
    # plt.savefig("fig/knn_unknown.png")
    # plt.show()
