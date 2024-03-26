from flask import Flask, request, jsonify, render_template
import numpy as np
import cv2
from keras.models import load_model

model = load_model('./CNN_ML_Model.keras')

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    try:
        image = preprocess_image(file)
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        class_labels = ['Bicycle', 'Car', 'Deer', 'Mountain']
        max_index = np.argmax(prediction[0])
        prediction_str = class_labels[max_index]
        return jsonify({'prediction': prediction_str})
    except Exception as e:
        return jsonify({'error', str(e) }), 500
    
def preprocess_image(uploaded_file, targer_size=(224,224)):
    image_stream = uploaded_file.read()
    image_stream = np.frombuffer(image_stream, np.uint8)
    image = cv2.imdecode(image_stream, cv2.IMREAD_COLOR)
    image = cv2.resize(image, targer_size)
    image = image / 255.0
    return image

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

"""
the allowed_file function checks if a given filename has an allowed file extension by looking for a period in the filename, 
extracting the extension, converting it to lowercase, and checking if it's in a set of predefined allowed extensions
"""

if __name__ == '__main__':
    app.run(debug=False)


    """ 
     
      
Flask Route for Prediction
@app.route('/predict', methods=['POST']): This decorator defines the route /predict that listens for HTTP POST requests. The methods parameter specifies that this route only accepts POST requests.
def predict(): This function is executed when a POST request is made to the /predict endpoint.
The function first checks if a file was uploaded with the request. If no file is found, it returns a JSON response with an error message and a 400 status code, indicating a bad request.
It then checks if the uploaded file is of an allowed type using the allowed_file function. If the file type is not allowed, it returns a JSON response with an error message and a 400 status code.
The function then attempts to preprocess the uploaded image using the preprocess_image function, make a prediction using a machine learning model, and return the prediction result as a JSON response.
If an exception occurs during the file processing or prediction, it returns a JSON response with an error message and a 500 status code, indicating a server error.
Preprocessing Function
def preprocess_image(uploaded_file, targer_size=(224,224)): This function takes an uploaded file and a target size for the image. It reads the file, decodes it into an image using OpenCV, resizes the image to the target size, normalizes the pixel values, and returns the preprocessed image.
Allowed File Type Check
def allowed_file(filename): This function checks if the uploaded file's extension is in a list of allowed extensions (png, jpg, jpeg). 
It returns True if the file type is allowed, False otherwise.

Key Libraries and Concepts:

Flask: A web framework for Python that allows for the development of web applications.
NumPy: A library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
OpenCV (cv2): A library of programming functions mainly aimed at real-time computer vision.
Machine Learning Model: A model trained to make predictions based on input data. The specific model used in this code is not detailed, but it's assumed to be a pre-trained model capable of classifying images into categories such as 'Bicycle', 'Car', 'Deer', and 'Mountain'.   
        
     """