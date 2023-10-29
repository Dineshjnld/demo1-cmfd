import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

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

st.title("Copy-Move Forgery Detection")
st.text(' Its for demo purpose')
st.text('582/5a1/595/569')
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Read the uploaded image using BytesIO
    image_data = uploaded_image.read()
    
    # Open the image with Pillow
    uploaded_image = Image.open(BytesIO(image_data))
    
    # Preprocess the uploaded image
    preprocessed_image = preprocess_uploaded_image(uploaded_image)
    
    # Make a prediction
    preprocessed_image = preprocessed_image.reshape(1, 224, 224, 3)
    prediction = model.predict(preprocessed_image)
    positive_class_probability = prediction[0, 1]
    
    # Display the result
    st.write("The Image is ", predict_image(positive_class_probability))
    st.write("Authenticity Probability:", positive_class_probability)
