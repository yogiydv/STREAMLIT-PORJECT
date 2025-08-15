import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load model
MODEL_PATH = "alzheimers_model.h5"
model = load_model(MODEL_PATH)

# UI
st.title("🧠 Alzheimer's Disease Detection from MRI Scans")
st.write("Upload an MRI scan and the model will predict the stage of Alzheimer's.")

# Upload image
uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

# Prediction function
def predict_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100
    return predicted_class, confidence

# Run prediction
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", use_column_width=True)
    st.write("Processing...")
    label, conf = predict_image(image)
    st.success(f"Prediction: {label} ({conf:.2f}% confidence)")
