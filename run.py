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
    # if pred == Label.Left:
    #     keyboard.Controller().press(keyboard.Key.space)
    #     keyboard.Controller().release(keyboard.Key.space)
    # elif pred == Label.Right:
    #     keyboard.Controller().press(keyboard.Key.backspace)
    #     keyboard.Controller().release(keyboard.Key.backspace)
    # elif pred == Label.Fist:
    #     keyboard.Controller().press(keyboard.Key.enter)
    #     keyboard.Controller().release(keyboard.Key.enter)
    # elif pred == Label.Paper:
    #     keyboard.Controller().press(keyboard.KeyCode.from_char("a"))
    #     keyboard.Controller().release(keyboard.KeyCode.from_char("a"))
    # elif pred == Label.Spider:
    #     keyboard.Controller().press(keyboard.KeyCode.from_char("b"))
    #     keyboard.Controller().release(keyboard.KeyCode.from_char("b"))

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
