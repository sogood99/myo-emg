from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from data import label_to_str

from pathlib import Path

sample_path = Path("data") / "samples.npy"
label_path = Path("data") / "labels.npy"

X = np.load(sample_path)
y = np.load(label_path).squeeze()

y_label = []

for i in range(len(y)):
    y_label.append(label_to_str(y[i]))

model, _ = torch.load("model.pth")
model.eval()

X = torch.tensor(X, dtype=torch.float32)

print(model)
# first 2 layers
model = model[:3]
print(model)

X = model(X).detach().numpy()


X_embedded = TSNE(n_components=2).fit_transform(X)

for i in range(6):
    X_i = X_embedded[y == i]
    plt.scatter(X_i[:, 0], X_i[:, 1], label=label_to_str(i))
plt.legend()
plt.show()
