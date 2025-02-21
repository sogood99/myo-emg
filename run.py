# Simplistic data recording
import time
import multiprocessing
import numpy as np
from data import Label
from knn import Classifier
import torch

from data import label_to_str

from pyomyo import Myo, emg_mode

from pynput import keyboard, mouse


def handle_prediction(pred):
    # if pred == Label.Fist:
    #     mouse.Controller().move(-100, 0)
    # elif pred == Label.Paper:
    #     mouse.Controller().move(100, 0)
    # elif pred == Label.HalfHeart:
    #     mouse.Controller().move(0, -100)
    # elif pred == Label.Zero:
    #     mouse.Controller().move(0, 100)
    # elif pred == Label.Gun:
    #     mouse.Controller().click(mouse.Button.left)
    print(label_to_str(pred))


def run_predictor(seconds, mode):
    """
    Run the predictor for a given number of seconds
    """

    cls = Classifier(
        np.load("data/samples.npy"),
        np.load("data/labels.npy").squeeze(),
        torch.load("model.pth", weights_only=False)[0][:3],
    )

    m = Myo(mode=mode)
    m.connect()

    myo_data = []

    def add_to_queue(emg, movement):
        myo_data.append(emg)

    m.add_emg_handler(add_to_queue)

    def print_battery(bat):
        print("Battery level:", bat)

    m.add_battery_handler(print_battery)

    m.set_leds([0, 128, 0], [0, 128, 0])
    m.vibrate(1)

    print("Started running predictor")
    start_time = time.time()

    while True:
        if time.time() - start_time < seconds:
            m.run()
            if len(myo_data) > 25:
                data = np.array(myo_data[:25])
                data = data.reshape(1, 25, 8)
                data = data.mean(axis=1)
                data = data.reshape(1, -1)
                pred = cls.predict(data)[0]
                if pred != Label.Unknown:
                    print(pred)
                    handle_prediction(pred)
                myo_data = myo_data[25:]

        else:
            break


if __name__ == "__main__":
    seconds = 10000
    mode = emg_mode.PREPROCESSED
    run_predictor(seconds, mode)
