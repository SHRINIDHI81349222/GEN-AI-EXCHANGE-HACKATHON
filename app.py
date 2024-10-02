from flask import Flask, request, jsonify
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model (provide the path to your .h5 model file)
model = tf.keras.models.load_model('C:/Users/nidhi/lung_cancer_detector.h5')

# Ensure the 'uploads' folder exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Function to process and predict the image
def predict_image(image_path):
    # Load and process the image
    img = image.load_img(image_path, target_size=(150, 150))  # Adjust the size if needed
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    # Make a prediction
    prediction = model.predict(img_array)
    
    # Return the prediction (you can customize the format)
    return prediction

# Define the API endpoint to handle file uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save the uploaded file to the 'uploads' directory
    filepath = os.path.join('uploads', file.filename)
    file.save(filepath)
    
    # Predict the image using the AI model
    prediction = predict_image(filepath)
    
    # Example: Convert prediction to a simple classification
    predicted_class = 'Cancerous' if prediction[0][0] > 0.5 else 'Non-cancerous'
    
    # Return the prediction as JSON
    return jsonify({'prediction': predicted_class})
@app.route('/')
def home():
    return "Welcome to the Lung Cancer Detection App"


# Start the Flask server
if __name__ == '__main__':
    app.run(debug=True)
