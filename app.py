from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
import base64

app = Flask(__name__)
CORS(app)  # Allow frontend requests

# Load your trained model
model = tf.keras.models.load_model("models/gesture_recognition_model.h5")


# Define gesture labels (based on your dataset)
gesture_labels = ['palm', 'L', 'fist', 'fist_moved', 'thumb', 'index', 'ok', 'palm_moved', 'C', 'down']

@app.route("/")
def home():
    return "Flask API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["image"]  # Get image from React frontend (Base64)
        image_data = base64.b64decode(data)
        np_arr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Preprocess image (resize to match model input)
        img = cv2.resize(img, (64, 64))  # Adjust based on model input size
        img = img / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict using model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction)

        # Return predicted gesture
        return jsonify({"gesture": gesture_labels[predicted_class]})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
