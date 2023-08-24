from flask import Flask, render_template, request, redirect, jsonify
from tensorflow.keras.models import load_model
import PIL
import numpy as np


#Initialize flask app
app = Flask(__name__)

#load your pretrained model
model = load_model('imageClassifier/Project/models/CNN.h5')

def predict_digit(digit_image):
    #reshape image
    digit_image = digit_image.reshape(-1, 28, 28, 1)
    
    #predict digit using model
    pred_digit = model.predict(digit_image)
    
    return pred_digit.argmax()

def load_image(image):
    #load image
    img = PIL.Image.open(image)
    
    
    #convert image to array
    img = (np.array(img))
    
    #reshaping to support our model input and normalizing
    img = img.reshape(-1, 28, 28, 1)
    
    return img    
print(model.predict((np.log1p(np.array(load_image('imageClassifier/Project/saved_images/image_431.png'))))).argmax())