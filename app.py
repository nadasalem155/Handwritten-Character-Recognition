import streamlit as st

# Page config
st.set_page_config(page_title="Handwritten Text Recognizer", page_icon="✍️", layout="centered")

import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# Simple model loading
st.write("Loading model...")
model = load_model("handwritten_cnn_model.keras")
st.write("Model loaded successfully!")

# Cache the label mapping
@st.cache_data
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    unique_labels = sorted(list(set(df['label'])))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return label_to_index, index_to_label

label_to_index, index_to_label = load_label_mapping("english.csv")
num_classes = len(index_to_label)

# Custom CSS
st.markdown("""
    <style>
    .stApp { background-color: #f0f2f6; font-family: 'Arial', sans-serif; }
    .title { font-size: 48px; font-weight: bold; color: #1a1a1a; text-align: center; margin-bottom: 10px; }
    .subtitle { font-size: 20px; color: #555; text-align: center; margin-bottom: 30px; }
    .stButton>button { background-color: #6200ea; color: white; font-size: 18px; font-weight: bold; padding: 10px 20px; border-radius: 8px; border: none; transition: background-color 0.3s; }
    .stButton>button:hover { background-color: #3700b3; }
    .prediction-box { background: linear-gradient(135deg, #bbdefb, #9575cd); border-radius: 15px; padding: 30px; text-align: center; color: #1a1a1a; font-size: 56px; font-weight: bold; box-shadow: 0 4px 15px rgba(0,0,0,0.2); margin-left: 20px; }
    </style>
""", unsafe_allow_html=True)

# Title and subtitle
st.markdown('<div class="title">✍️ Handwritten Text Recognizer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a single character on the canvas and click Predict to see the result!</div>', unsafe_allow_html=True)

# Columns: canvas and prediction side by side
col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="white", stroke_width=15, stroke_color="black", background_color="white",
        width=280, height=280, drawing_mode="freedraw", key="canvas", display_toolbar=True
    )

with col2:
    st.write("")
    if st.button("✨ Predict"):
        if canvas_result.image_data is not None:
            with st.spinner("Predicting..."):
                img = Image.fromarray(canvas_result.image_data.astype("uint8"))
                img = ImageOps.grayscale(img)
                img = img.resize((32, 32))
                img_array = np.array(img).astype("float32") / 255.0
                img_array = img_array.reshape(1, 32, 32, 1)
                prediction = model.predict(img_array)
                pred_class = np.argmax(prediction, axis=1)[0]
                pred_label = index_to_label[pred_class]
                st.markdown(f'<div class="prediction-box">Predicted Character: {pred_label}</div>', unsafe_allow_html=True)
        else:
            st.warning("Please draw a character before predicting.")