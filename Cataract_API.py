import fastapi
import uvicorn
import cv2
import tensorflow as tf
import numpy as np
import time
from utils import *
import efficientnet.tfkeras
from classification_models.tfkeras import  Classifiers

# Change these paths accordingly
resnext_path = 'Cataract_Resnext50.h5'
resnet_path = 'Cataract_Resnet50.h5'
effnet_path = 'Cataract_EfficientNetB0.h5'
inception_path = 'Cataract_InceptionV3.h5'

# Loading the models
model_resnext = model_load(resnext_path)
model_resnet = model_load(resnet_path)
# model_inception = tf.keras.models.load_model(inception_path)
model_effnet = model_load(effnet_path)


# API code starts here :
app = fastapi.FastAPI()
@app.get("/")
def read_root():
    return {"Main Page":"1"}


@app.get("/cataract/{loc}")
def cataract(loc:str):
    start = time.time()

    im = cv2.imread("/home/amokh/Desktop/"+loc+".png")

    if(im is None):
    	return {"1" : "Not Defined" , 'time' : "Error"}


    prediction_effnet = predict_effnet(model_effnet,im)
    prediction_resnet = predict_resnet(model_resnet,im)
    prediction_resnext = predict_resnext(model_resnext,im)
    #prediction_inception = predict_inception(model_inception,im)
    prediction = (prediction_effnet + prediction_resnet + prediction_resnext)/3
    end = time.time()
    t = end-start
    return {"1":str(prediction[0]),'time':str(t)}


uvicorn.run(app, host="0.0.0.0", port=8000)
