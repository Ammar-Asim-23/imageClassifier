import numpy as np
import PIL.Image
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the pretrained CNN model
model = load_model('imageClassifier/Project/models/CNN.h5')

# Preprocess the image using the provided load_image function
def load_image(image):
    img = PIL.Image.open(image)
    img = np.array(img)
    img = img.reshape(-1, 28, 28, 1)
    return img

# Predict digit using the provided predict_digit function
def predict_digit(digit_image):
    pred_digit = model.predict(digit_image)
    return pred_digit.argmax()

@app.route('/predict', methods=['POST'])
def predict_digit_endpoint():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        # Load the image using the load_image function
        processed_image = load_image(image_file)
        # Predict the digit using the predict_digit function
        predicted_digit = predict_digit(processed_image)
        
        response = {'predicted_digit': int(predicted_digit)}
        return jsonify(response)
    
    except Exception as e:
        error_response = {'error': str(e)}
        return jsonify(error_response), 400

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
