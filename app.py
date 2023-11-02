import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from io import BytesIO

model = load_model("model1.h5")

def preprocess_uploaded_image(uploaded_image):
    uploaded_image = cv2.resize(uploaded_image, (224, 224))
    uploaded_image = uploaded_image.astype(np.float32) / 255.0
    return uploaded_image

def predict_image(authenticity_probability):
    threshold = 0.5
    if authenticity_probability > threshold:
        return "Authentic"
    else:
        return "Forgery"

st.title("Copy-Move Forgery Detection")


uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Read the uploaded image using BytesIO
    image_data = uploaded_image.read()
    
    # Convert image data to a NumPy array
    nparr = np.frombuffer(image_data, np.uint8)
    
    # Decode the NumPy array to an OpenCV image
    uploaded_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Preprocess the uploaded image
    preprocessed_image = preprocess_uploaded_image(uploaded_image)
    
    # Make a prediction
    preprocessed_image = preprocessed_image.reshape(1, 224, 224, 3)
    prediction = model.predict(preprocessed_image)
    positive_class_probability = prediction[0, 1]
    
    # Display the result
    st.write("The Image is ", predict_image(positive_class_probability))
    st.write("Authenticity Probability: {:.4f}".format(positive_class_probability))
