from keras.models import Model, load_model
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from scipy.misc import imread, imresize
import numpy as np
import csv

class Predict:
    '''load the model and the label names
       model_path: string contaning path to model.h5 file
       label_path: string contaning path to label.csv file
       Returns: None'''
    def __init__(self, model_path, label_path):
        self.my_model = None

        # load the model
        self.my_model = load_model(model_path)
        
        # load labels
        self.label_names = {}
        with open(label_path) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self.label_names[int(line[1])] = line[0]

    '''pass the input to the model for prediction
       Returns: list containing top 3 predictions along with percentage probability'''
    def make_prediction(self, np_img):
        x = np.expand_dims(np_img, axis=0)
        preds = self.my_model.predict(x)
        idx = np.argsort(preds)[0][-3:][::-1]
        percent = preds[0][idx]*100
        out = [self.label_names[i] for i in idx]        
        return list(zip(out,percent))

    '''load the image from the given path and convert it to numpy array
       Returns: numpy array'''
    def load_img_from_path(self, image_path):
        x = imresize(imread(image_path, mode='RGB'), (224, 224)).astype(np.float32)
        x = image.img_to_array(x)
        return x





   