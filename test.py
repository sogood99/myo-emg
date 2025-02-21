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

test_data = os.listdir("test_data")

X = []
y = []

for file in test_data:
    df = pd.read_csv(Path("test_data") / file)
    X.append(df.to_numpy())
    label = str_to_label(file)
    y += [label] * len

    print(df)
