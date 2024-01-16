from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)


model = load_model('candy_classifier_model.h5')

class_labels = {0: 'kitkat', 1: 'magobite', 2: 'skittles'}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['image']

    # Save the file to a temporary location
    file_path = 'test.jpg'
    file.save(file_path)

    # Load the saved image and preprocess it
    img = image.load_img(file_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = class_labels.get(predicted_class, 'Unknown')

    # Return the result as JSON
    result = {'prediction': predicted_label}
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
