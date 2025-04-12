import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Load your trained model
model = load_model('best_model.keras')

# Class labels from your dataset
class_names = ['Abnormal_Sperm', 'Non-Sperm', 'Normal_Sperm']

st.set_page_config(page_title="Sperm Cell Classifier", layout="centered")
st.title("🧬 Sperm Cell Classification")
st.markdown("Upload a sperm sample image and get a prediction.")

# Upload image
uploaded_file = st.file_uploader("Choose a sperm cell image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    image_pil = Image.open(uploaded_file)
    st.image(image_pil, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img = image_pil.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    # Show prediction
    st.success(f"### Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence * 100:.2f}%")

# Optionally show confusion matrix
if st.checkbox("Show Training Confusion Matrix"):
    st.subheader("Confusion Matrix - Training Data")
    st.image("confusion_matrix_train.png", caption="Confusion Matrix")

# Optionally show accuracy/loss plot
if st.checkbox("Show Accuracy and Loss Plot"):
    st.subheader("Model Training Performance")
    st.image("accuracy_loss_plot.png", caption="Accuracy & Loss")
