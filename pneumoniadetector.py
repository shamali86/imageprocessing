import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import pyttsx3

# Set Title
st.title('Pneumonia Detection App')

# Set header
st.header('Please upload the chest x-ray image to detect')

# upload the file
uploaded_file = st.file_uploader('', type=['jpeg', 'jpg'])

# load the classifier
model = load_model(r'D:\Vacasion_Upgrade_Skills\VideoClassification\pneumonia_detector\pneumonia_detector.h5')

# Function to preprocess image
def preprocess_image(image):
    # Resize the image to the required input size
    image = image.resize((150, 150))  # Adjust size as needed
    if image.mode != 'L':  # Convert to grayscale if not already
        image = image.convert('L')  # Convert to grayscale

    image = image.convert('RGB')  # Convert grayscale to RGB
    # Convert the image to a numpy array
    image = img_to_array(image)
    # Expand dimensions to include batch size and channels
    image = np.expand_dims(image, axis=0)  # Shape becomes (1, 150, 150, 3)
    # Normalize the image data
    image = image.astype('float32') / 255.0
    return image

# Function to predict pneumonia
def predict_pneumonia(image):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)[0][0]
    return prediction

# Function for text-to-speech
def speak_output(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if uploaded_file is not None:
    # Display the image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Chest X-ray", use_column_width=True)

    # Make a prediction
    prediction = predict_pneumonia(image)

    # Output result
    if prediction > 0.5:
        result = "Positive for Pneumonia"
    else:
        result = "Negative for Pneumonia"
    
    st.write(result)

    # Convert result to speech
    speak_output(result)