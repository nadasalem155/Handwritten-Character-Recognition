import streamlit as st

# Page configuration
st.set_page_config(page_title="Handwritten Character Recognition", page_icon="✍️", layout="centered")

import numpy as np
from PIL import Image, ImageOps
import pandas as pd
from tensorflow.keras.models import load_model
from streamlit_drawable_canvas import st_canvas

# -----------------------------
# Load model
# -----------------------------
model = load_model("handwritten_cnn_model.keras", compile=False)

# -----------------------------
# Cache label mapping
# -----------------------------
@st.cache_data
def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    unique_labels = sorted(list(set(df['label'])))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    return label_to_index, index_to_label

label_to_index, index_to_label = load_label_mapping("english.csv")
num_classes = len(index_to_label)

# -----------------------------
# Custom CSS for nicer UI
# -----------------------------
st.markdown("""
    <style>
    /* Background and fonts */
    .stApp { 
        background: linear-gradient(135deg, #0d0d0d, #1a1a1a); /* Dark gradient background */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
    }

    /* Title */
    .title { 
        font-size: 52px; 
        font-weight: bold; 
        color: #b39ddb; /* Light purple for contrast */
        text-align: center; 
        margin-bottom: 5px; 
        text-shadow: 1px 1px 3px rgba(255,255,255,0.2);
    }

    /* Subtitle */
    .subtitle { 
        font-size: 22px; 
        color: #d1c4e9; /* Soft purple for contrast */
        text-align: center; 
        margin-bottom: 40px; 
    }

    /* Buttons */
    .stButton>button { 
        background-color: #4a148c; 
        color: white; 
        font-size: 18px; 
        font-weight: bold; 
        padding: 12px 28px; 
        border-radius: 12px; 
        border: none; 
        transition: background-color 0.3s, transform 0.2s; 
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
    }
    .stButton>button:hover { 
        background-color: #7b1fa2; 
        transform: translateY(-2px);
        box-shadow: 0 6px 14px rgba(0,0,0,0.25);
    }

    /* Prediction box - matching title color */
    .prediction-box { 
        background: linear-gradient(135deg, #b39ddb, #b39ddb); /* Matching title color */
        border-radius: 20px; 
        padding: 10px 20px; /* Reduced padding for tighter fit */
        text-align: center; 
        white-space: nowrap;
        color: #1a1a1a; /* Dark text for contrast */
        font-size: 48px; 
        font-weight: bold; 
        box-shadow: 0 6px 20px rgba(0,0,0,0.25); 
        display: inline-block; /* Allows box to fit content */
        margin-top: 20px;
        transition: transform 0.3s;
    }
    .prediction-box:hover {
        transform: scale(1.03);
    }

    /* Canvas - white background */
    .stCanvas>canvas {
        border: 4px solid #7b1fa2;
        border-radius: 20px;
        background-color: #ffffff !important;  /* White background for canvas */
        width: 380px !important;  /* Ensure canvas width is fixed */
        height: 280px !important; /* Ensure canvas height is fixed */
    }
    .stCanvas {
        display: inline-block !important; /* Prevent extra space */
        background-color: transparent !important; /* Remove any unintended background */
    }

    /* Layout adjustment for canvas and predict button */
    .canvas-container {
        display: flex;
        align-items: center;
        gap: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Title and subtitle
# -----------------------------
st.markdown('<div class="title">✍️ Handwritten Character Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Draw a single character on the canvas and click Predict to see the result!</div>', unsafe_allow_html=True)

# -----------------------------
# Canvas and Predict button layout
# -----------------------------
col1, col2 = st.columns([1, 0.5])
with col1:
    st.markdown('<div class="canvas-container">', unsafe_allow_html=True)
    canvas_result = st_canvas(
        fill_color=None,            # transparent inside
        stroke_width=15, 
        stroke_color="black",      # Black stroke for visibility on white background
        background_color="#ffffff", # White background
        width=380, 
        height=280, 
        drawing_mode="freedraw", 
        key="canvas", 
        display_toolbar=True
    )
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    if st.button("✨ Predict"):
        if canvas_result.image_data is not None:
            # Check if any drawing exists
            img = Image.fromarray(canvas_result.image_data.astype("uint8"))
            img_array = np.array(img).astype("float32") / 255.0
            if np.all(img_array == 1.0):  # Check if canvas is still all white
                st.warning("No input drawn. Please draw a character before predicting.")
            else:
                with st.spinner("Predicting..."):
                    img = ImageOps.grayscale(img)
                    img = img.resize((32, 32))
                    img_array = np.array(img).astype("float32") / 255.0
                    img_array = img_array.reshape(1, 32, 32, 1)
                    prediction = model.predict(img_array)
                    pred_class = np.argmax(prediction, axis=1)[0]
                    pred_label = index_to_label[pred_class]
                    st.markdown(f'<div class="prediction-box">Predicted Character: {pred_label}</div>', unsafe_allow_html=True)
        else:
            st.warning("No input drawn. Please draw a character before predicting.")