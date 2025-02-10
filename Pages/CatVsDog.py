import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Constants
CATEGORIES = ['cat', 'dog']
IMG_WIDTH = 254
IMG_HEIGHT = 254

# Load the compressed TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="Pages/Models/model_compressed.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Image preprocessing function
def preprocess_image(uploaded_file):
    """
    Preprocess a single image uploaded by the user.
    """
    # Read the uploaded file as bytes and decode it to a NumPy array
    img_array = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)  # Decode the image
    
    # Check if the image was successfully loaded
    if img is None:
        raise ValueError("Failed to load the image. Ensure it's a valid image file.")
    
    # Resize the image and normalize it
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img / 255.0  # Normalize to [0, 1]
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img.astype(np.float32)


# Prediction function
def predict_single_image(image):
    """
    Predict the class of a single image using the TFLite model.
    """
    # Set the model's input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor (prediction score)
    prediction_score = interpreter.get_tensor(output_details[0]['index'])[0][0]
    return prediction_score


# Classifier function
def classifier(prediction_score):
    """
    Convert the raw prediction score to a class label.
    """
    if prediction_score < 0.5:
        return CATEGORIES[0]  # 'cats'
    else:
        return CATEGORIES[1]  # 'dogs'


# Streamlit App


image = Image.open('Images/CatVsDog/CatANDdog.png')
st.image(image,width=200)

st.title("Cat and Dog Classifier")

# Upload Image
uploaded_file = st.file_uploader("Upload an image of a cat or dog", type=["jpeg", "jpg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        image = preprocess_image(uploaded_file)
        
        # Get prediction
        prediction_score = predict_single_image(image)
        predicted_class = classifier(prediction_score)
        
        # Display results
 
        st.title("Prediction of Trained Model ")

        col1, col2 = st.columns(2)

        with col1:
            st.image(uploaded_file,width=200)

        with col2:
            st.markdown(f"<h3 style='text-align: center; color: green;'>Predicted Class: {predicted_class.capitalize()}</h3>", unsafe_allow_html=True)
            # st.markdown(f"<h4 style='text-align: center;'>Prediction Score: {prediction_score:.2f}</h4>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error: {e}")



