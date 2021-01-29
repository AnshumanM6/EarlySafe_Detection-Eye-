import cv2
import tensorflow as tf
import numpy as np
from classification_models.tfkeras import  Classifiers
from efficientnet import preprocessing
import tensorflow as tf

# Loads the Required model
def model_load(path):
    model = tf.keras.models.load_model(path)
    return model

# The following functions give the out[ut predictions for the models

def predict_resnext(model,image):
    _, preprocess_resnext = Classifiers.get('resnext50')
    im = cv2.resize(image,(224,224))
    print(im.shape)
    im = np.reshape(im,(-1,224,224,3))
    im = preprocess_resnext(im)
    prediction = model.predict(im)
    return prediction


def predict_resnet(model,image):
    _, preprocess_resnet = Classifiers.get('resnet50')
    im = cv2.resize(image,(224,224))
    im = np.reshape(im,(-1,224,224,3))
    im = preprocess_resnet(im)
    prediction = model.predict(im)
    return prediction

def predict_effnet(model,image):
    im = cv2.resize(image,(300,300))
    im = preprocessing.center_crop_and_resize(im,300)
    im = np.reshape(im,(-1,300,300,3))
    prediction = model.predict(im)
    return prediction

def predict_inception(model,image):
    im = cv2.resize(image,(300,300))
    im = im/255
    im = np.reshape(im,(-1,300,300,3))
    prediction = model.predict(im)
    return prediction