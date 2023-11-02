import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO
import os

model = load_model("model1.h5")

def preprocess_uploaded_image(uploaded_image):
    uploaded_image = uploaded_image.resize((224, 224))
    uploaded_image = np.array(uploaded_image) / 255.0
    return uploaded_image

def predict_image(authenticity_probability):
    threshold = 0.5
    if authenticity_probability > threshold:
        return "Authentic"
    else:
        return "Forgery"

# Sample images that users can download
sample_images = {
    "Sample 1": "./demos/sample1.jpg",
    "Sample 2": "./demos/sample2.jpg",
    "Sample 3": "./demos/sample3.jpg"
}

st.title("Copy-Move Forgery Detection")
st.text("It's for demo purposes")
st.text("582/5a1/595/569")

# Add links for downloading sample images
st.sidebar.write("Download Sample Images:")
for sample_name, sample_image_path in sample_images.items():
    st.sidebar.markdown(f"- [{sample_name}]({sample_image_path})")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    image_data = uploaded_image.read()
    
    uploaded_image = Image.open(BytesIO(image_data))
    
    preprocessed_image = preprocess_uploaded_image(uploaded_image)
    
    preprocessed_image = preprocessed_image.reshape(1, 224, 224, 3)
    prediction = model.predict(preprocessed_image)
    positive_class_probability = prediction[0, 1]
    
    st.write("The Image is ", predict_image(positive_class_probability))
    st.write("Authenticity Probability:", positive_class_probability)
    st.write("hi")
