import csv
import os
import time

import numpy as np
from keras.layers import (Activation, AveragePooling2D, Convolution2D, Dense,
                          Dropout, Flatten, Input, MaxPooling2D, Reshape,
                          ZeroPadding2D, merge)
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.utils import np_utils
from scipy.misc import imread, imresize

import googlenet_custom_layers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# load the pretrained model
googlenet = load_model('googlenet.h5', custom_objects={
                       'PoolHelper': googlenet_custom_layers.PoolHelper, 'LRN': googlenet_custom_layers.LRN})
# googlenet.summary()

num_classes = 38
X_train = []
Y_train = []
csv_data = []

# read the data in csv file
with open('./plant_data.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        csv_data.append(line)
    
print(csv_data[0])

sklearn.utils.shuffle(csv_data)
train_data, validation_data = train_test_split(csv_data, test_size=0.2)

def generator(csv_data, batch_size = 28):
    data_length = len(csv_data)
    while 1:
        for offset in range(0, data_length, batch_size):
            # split the data into batches
            batch_data = csv_data[offset:offset+batch_size]
            images = []
            labels = []

            for data in batch_data:
                image_path = './data/' + data[0]
                img = imresize(imread(image_path, mode='RGB'), (224, 224)).astype(np.float32)
                img = img.transpose((2, 0, 1))
                images.append(img)
                labels.append(int(data[2]))

            X_train = np.array(images)
            Y_train = np.array(labels)
            Y_train = np_utils.to_categorical(labels, num_classes)
            yield sklearn.utils.shuffle(X_train, Y_train)

gen_instance = generator(train_data)
validation_generator = generator(validation_data, batch_size=32)

#########################################################################################


flattened_layer = googlenet.get_layer('flatten_3').output
fc_1 = Dense(512, activation='relu', name='fc_1')(flattened_layer)
output = Dense(num_classes, activation='softmax', name='output')(fc_1)

image_input = googlenet.get_layer('input_1').input
my_googlenet = Model(image_input, output)

for layer in my_googlenet.layers[:-2]:
    layer.trainable = False

my_googlenet.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

t=time.time()
hist = my_googlenet.fit_generator(generator(train_data), steps_per_epoch = len(train_data)//28, epochs = 1, validation_data=generator(validation_data), validation_steps=len(validation_data)//28)
print('Training time: %s' % (t - time.time()))
(loss, accuracy) = my_googlenet.evaluate(X_test, y_test, batch_size=10, verbose=1)

print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,accuracy * 100))

