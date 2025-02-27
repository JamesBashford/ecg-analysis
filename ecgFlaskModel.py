from flask import Flask, request, jsonify
import gdown
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask("ECG Analysis")

# Google Drive file ID for the ECG model
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

# Updated function to handle ANY image by forcing it to RGB
def preprocess_image(image):
    # 1. Open the file and convert to RGB, ensuring 3 channels (color)
    img = Image.open(image).convert('RGB')
    
    # 2. Resize to match your model’s expected input size
    img = img.resize((128, 128))
    
    # 3. Convert to NumPy and normalize if needed
    img_array = np.array(img).astype(np.float32) / 255.0
    
    # 4. Add batch dimension → shape: (1, 128, 128, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the file from the request
        file = request.files['file']
        
        # Preprocess the image (now guaranteed to be 3 channels)
        img_array = preprocess_image(file)
        
        # Make a prediction
        prediction = model.predict(img_array)[0][0]  # get the first (and only) prediction
        
        # Convert prediction to class label
        # Here we assume a binary model output in range 0..1
        predicted_class = classes[int(prediction > 0.5)]
        confidence = float(prediction if predicted_class == 'Heart Attack' else 1 - prediction)
        
        # Return the result as JSON
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": confidence
        })
    except Exception as e:
        print(f"Error occurred: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/')
def index():
    return "Welcome to the ECG Analysis API. Use the /predict endpoint to analyze images."

if __name__ == '__main__':
    # Use the dynamically assigned port from the environment, default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
