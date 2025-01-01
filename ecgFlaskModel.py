from flask import Flask, request, jsonify
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask("ECG Analysis")

# Google Drive file ID for your ECG model
FILE_ID = '1CZlPW-Lpoao4Hm-MD7HHSnCJxNLdL-ay'

# Correct URL for the Google Drive file
url = f'https://drive.google.com/uc?id={FILE_ID}'

# Local path where the model will be saved
model_path = 'ecg_tf_model.h5'

# Download the model from Google Drive if it doesn't exist
if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

# Load the model
model = load_model(model_path)

# Class labels
classes = ['Normal', 'Heart Attack']

# Function to preprocess an image
def preprocess_image(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))  # Resize to match model input
    img_array = np.array(img).astype(np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = np.expand_dims(img_array, axis=-1)  # Add channel dimension
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request
        file = request.files['file']
        
        # Preprocess the image
        img_array = preprocess_image(file)
        
        # Make a prediction
        prediction = model.predict(img_array)[0][0]  # Get the first (and only) prediction
        
        # Convert prediction to class label
        predicted_class = classes[int(prediction > 0.5)]  # Threshold: 0.5
        confidence = float(prediction if predicted_class == 'Heart Attack' else 1 - prediction)
        
        # Return the result as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

# Optional: Add a route for the root URL
@app.route('/')
def index():
    return "Welcome to the ECG Analysis API. Use the /predict endpoint to analyze images."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
