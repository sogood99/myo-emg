from knn import Classifier
import numpy as np
import torch
import os
import pandas as pd
from pathlib import Path

from data import *

cls = Classifier(
    np.load("data/samples.npy"),
    np.load("data/labels.npy").squeeze(),
    torch.load("model.pth", weights_only=False)[0][:3],
)

X_test = np.load("test_data/samples.npy")
y_test = np.load("test_data/labels.npy").squeeze()

y_pred = cls.predict(X_test)

accuracy = np.mean(y_pred == y_test)
print(accuracy)
