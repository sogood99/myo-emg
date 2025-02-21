import matplotlib.pyplot as plt
import matplotlib.cm
import os
import pandas as pd
from pathlib import Path


data = os.listdir("data")


def plot_data(data):
    # plot the 8 channels
    cm = plt.get_cmap("tab10")
    for i in range(8):
        plt.subplot(8, 1, i + 1)
        plt.ylim(0, 1500)
        plt.plot(data.iloc[:, i], color=cm(i), label=f"Channel {i + 1}")
        plt.title(f"Channel {i + 1}")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.6)


if __name__ == "__main__":

    for d in data:
        df = pd.read_csv(Path("data") / d)

        plt.cla()
        plt.figure(figsize=(8, 12))
        plot_data(df)
        plt.savefig(Path("plots") / f"{d}.png")
        # plt.show()
        print(df)
