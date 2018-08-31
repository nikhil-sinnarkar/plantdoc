from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import csv

class Predict:
    def __init__(self, model_path, label_path):
        # load the model
        my_model = load_model(model_path)
        