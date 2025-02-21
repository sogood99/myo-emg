import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split

from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

sample_path = Path("data") / "samples.npy"
label_path = Path("data") / "labels.npy"

X = np.load(sample_path)
y = np.load(label_path)

dim = X.shape[1]

model = nn.Sequential(
    nn.Linear(dim, 16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU(),
    nn.Linear(32, 6),
)
critereon = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long).squeeze()
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.long).squeeze()

print(X_train.shape, X_test.shape)

for epoch in range(1, 5001):
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = critereon(y_pred, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch} Loss: {loss.item()}")

    writer.add_scalar("Loss/train", loss.item(), epoch)

    if epoch % 100 == 0:
        # eval on test set
        with torch.no_grad():
            y_pred = model(X_test)
            loss = critereon(y_pred, y_test)
            print(f"Test Loss: {loss.item()}")

            writer.add_scalar("Loss/validation", loss.item(), epoch)

            # accuracy
            y_pred = torch.argmax(y_pred, dim=1)
            acc = (y_pred == y_test).sum().item() / len(y_test)
            print(f"Validation Accuracy: {acc}")
            writer.add_scalar("Accuracy/validation", acc, epoch)

            if not Path("model.pth").exists():
                torch.save((model, loss.item()), "model.pth")
                print("Model saved")

            prev_model, prev_loss = torch.load("model.pth", weights_only=False)
            if loss.item() < prev_loss:
                torch.save((model, loss.item()), "model.pth")
                print("Model saved")
            else:
                print("Model not saved")
