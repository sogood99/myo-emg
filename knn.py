from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import torch

k = 3

X = np.load("data/samples.npy")
y = np.load("data/labels.npy").squeeze()

X_unknown = np.load("data/unknown.npy")

model, _ = torch.load("model.pth")
model = model[:3]
model.eval()

X = torch.tensor(X, dtype=torch.float32)
X_transformed = model(X).detach().numpy()
X_unknown = torch.tensor(X_unknown, dtype=torch.float32)
X_unknown = model(X_unknown).detach().numpy()

knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_transformed, y)

distances = knn.kneighbors(X_transformed, n_neighbors=k, return_distance=True)

mean_distance = np.mean(distances[0], axis=1)
print(mean_distance)

threshold = np.percentile(mean_distance, 95)
print(threshold)

distances_unknown = knn.kneighbors(X_unknown, n_neighbors=k, return_distance=True)
print(distances_unknown)
