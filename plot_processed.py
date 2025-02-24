from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from data import label_to_str

from pathlib import Path


def plot_tsne(X, y):
    X_embedded = TSNE(n_components=2).fit_transform(X)

    for i in range(6):
        X_i = X_embedded[y == i]
        plt.scatter(X_i[:, 0], X_i[:, 1], label=label_to_str(i))
    plt.legend()


sample_path = Path("data") / "samples.npy"
label_path = Path("data") / "labels.npy"

X = np.load(sample_path)
y = np.load(label_path).squeeze()


plot_tsne(X, y)
plt.title("T-SNE Visualization of Data")
plt.savefig("fig/tsne_data.png")
plt.show()


model, _ = torch.load("model.pth", weights_only=False)
model.eval()

X = torch.tensor(X, dtype=torch.float32)

model = model[:3]

X = model(X).detach().numpy()

plot_tsne(X, y)
plt.title("T-SNE Visualization of Transformed Data")
plt.savefig("fig/tsne_transformed.png")
plt.show()
