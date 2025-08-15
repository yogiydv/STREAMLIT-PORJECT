
import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
import sys

MODEL_PATH = "model/alzheimer_model.h5"
DEFAULT_MODEL_URL = ""  # <-- Add your model download URL here (Google Drive direct link or Hugging Face raw link)

st.set_page_config(page_title="Alzheimer's Detection", layout='centered')
st.title("🧠 Alzheimer's Detection from MRI Scans")
st.write("Upload an MRI image to detect the stage of Alzheimer's disease. Educational demo only.")

def download_model(url, dest):
    if not url:
        return False, "No URL provided"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        # Support Google Drive (gdrive id) links via requests if direct raw link is provided
        resp = requests.get(url, stream=True, timeout=60)
        if resp.status_code == 200:
            with open(dest, "wb") as f:
                for chunk in resp.iter_content(1024*1024):
                    if chunk:
                        f.write(chunk)
            return True, "Downloaded"
        else:
            return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)

# If model not present, try downloading from Streamlit secrets or DEFAULT_MODEL_URL
if not os.path.exists(MODEL_PATH):
    model_url = st.secrets.get("MODEL_URL", "") if hasattr(st, "secrets") else ""
    if not model_url:
        model_url = DEFAULT_MODEL_URL
    if model_url:
        with st.spinner("Model not found locally — attempting download..."):
            ok, msg = download_model(model_url, MODEL_PATH)
            if ok:
                st.success("Model downloaded.")
            else:
                st.warning(f"Model download failed: {msg} — the app will run in demo mode.")
    else:
        st.info("No model URL configured. The app will run in demo mode.")

# Try to load model
MODEL = None
if os.path.exists(MODEL_PATH):
    try:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        MODEL = None

uploaded_file = st.file_uploader('Upload MRI Image', type=['jpg','jpeg','png'])
if uploaded_file:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, use_column_width=True)
    arr = np.array(img.resize((128,128))) / 255.0
    arr = np.expand_dims(arr, axis=0)
    if MODEL is not None:
        try:
            preds = MODEL.predict(arr)[0]
            classes = ['MildDemented','ModerateDemented','NonDemented','VeryMildDemented']
            idx = int(np.argmax(preds))
            st.success(f"Prediction: **{classes[idx]}** (Confidence: {preds[idx]*100:.2f}%)")
            st.bar_chart({c: float(p) for c,p in zip(classes, preds)})
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        # Demo heuristic: brightness-based
        gray = np.array(img.convert('L').resize((128,128))) / 255.0
        m = gray.mean()
        if m > 0.6:
            label='NonDemented'
        elif m > 0.45:
            label='VeryMildDemented'
        elif m > 0.3:
            label='MildDemented'
        else:
            label='ModerateDemented'
        st.info(f"Demo prediction: **{label}** (replace with a real model for accurate results)")        
