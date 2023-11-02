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

sample_images = {
    "Sample 1": "demos/sample1.jpg",
    "Sample 2": "demos/sample2.jpg",
    "Sample 3": "demos/sample3.jpg",
    "Sample 4": "demos/sample4.jpg",
    "Sample 5": "demos/sample5.jpg",
    "Sample 6": "demos/sample6.jpg",
}

st.title("Copy-Move Forgery Detection")
st.text(' Its for demo purpose')
st.text('582/5a1/595/569')

import streamlit as st

# with open("demos/sample2.jpg", "rb") as file:
#     btn = st.download_button(
#             label="sample image",
#             data=file,
#             file_name="demos/sample2.jpg",
#             mime="image/png"
#         )
with st.sidebar:
    st.markdown("## Sample Images")
    for sample_name, sample_path in sample_images.items():
        with open(sample_path, "rb") as file:
            # Include the file extension in the 'file_name' parameter
            file_extension = "jpg" if sample_path.endswith('.jpg') else "png"
            st.download_button(
                label=f"Download {sample_name}",
                data=file,
                file_name=f"{sample_name}.{file_extension}",
                mime="image/jpeg" if sample_path.endswith('.jpg') else "image/png"
            )
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
    st.write("Authenticity Probability:", (1-positive_class_probability))
    
