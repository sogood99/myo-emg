import numpy as np
import sklearn
import os
import pandas as pd
from pathlib import Path


class Label:
    Fist = 0
    Paper = 1
    Left = 2
    Right = 3
    Spider = 4
    Unknown = 5


def str_to_label(s):
    if s.startswith("fist"):
        return Label.Fist
    elif s.startswith("paper"):
        return Label.Paper
    elif s.startswith("left"):
        return Label.Left
    elif s.startswith("right"):
        return Label.Right
    elif s.startswith("spider"):
        return Label.Spider
    else:
        return Label.Unknown


def label_to_str(l):
    if l == Label.Fist:
        return "fist"
    elif l == Label.Paper:
        return "paper"
    elif l == Label.Left:
        return "Left"
    elif l == Label.Right:
        return "Right"
    elif l == Label.Spider:
        return "Spider"
    else:
        return "unknown"


if __name__ == "__main__":
    data = os.listdir("raw_data")

    X = []
    Y = []

    for d in data:
        df = pd.read_csv(Path("raw_data") / d)
        label = str_to_label(d)
        print(d, label)

        print("File: ", d)
        print("Label: ", label)

        dt = df.to_numpy()
        for i in range(500 - len(dt)):
            # expand the last row 3 times
            dt = np.insert(dt, len(dt), dt[-1], axis=0)
        dl = len(dt) // 20
        dt = dt.reshape(20, dl, 8)

        dt = dt.mean(axis=1)
        dt = dt.reshape(20, -1)
        y = np.full((20, 1), label)

        print(dt)

        if label == Label.Unknown:
            np.save(Path("data") / f"unknown.npy", dt)
        else:
            X += [dt]
            Y += [y]

    X = np.concatenate(X)
    Y = np.concatenate(Y)

    # pickle the data
    np.save(Path("data") / f"samples.npy", X)
    np.save(Path("data") / f"labels.npy", Y)
    print("Data saved")
    print(X.shape)
    print(Y.shape)
