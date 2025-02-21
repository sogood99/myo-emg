# Gesture Detection using MYO

## Project Structure

data/ - Contains the data files for training and testing the model
models/ - Contains the trained models

Data collected as 50hz, 8 channels, 10 seconds per gesture, 5 gestures (fist, paper, gun, half-heart, zero), preprocessed using low pass filter and FFT. Split data into chunks of 0.5 seconds (25 samples)
