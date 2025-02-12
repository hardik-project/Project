import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import math
import base64
import re
####################################################        ANN         ################################################################
######################################      Loading Datset      ###########################################

data = pd.read_csv('Csvs/Bikes/Used_Bikes.csv')
df = pd.DataFrame(data=data)

######################################      Encoding Datset      ###########################################

bike_encoder = LabelEncoder()
df['bike_id'] = bike_encoder.fit_transform(df['bike_name'])
city_encoder = LabelEncoder()
df['city_id'] = city_encoder.fit_transform(df['city'])
brand_encoder = LabelEncoder()
df['brand_id'] = brand_encoder.fit_transform(df['brand'])

replacement_dict = {
    'First Owner' : 1, 
    'Second Owner' : 2, 
    'Third Owner' : 3,
    'Fourth Owner Or More' : 4
}
df['owner_level'] = df['owner'].replace(replacement_dict)
def clean_model_name(model_name):
    return re.sub(r"\b\d+cc\b|\b\d+\b", "", model_name).strip()
    
df["cleaned_model"] = df["bike_name"].apply(clean_model_name)

####################################################        CNN         ################################################################
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
####################################################    streamlit app ########################################################################



# Sidebar menu for options
option = st.sidebar.selectbox("Choose a view:", ["Home","INFO", "ANN","CNN"])

if option == "Home":

    st.title("🚀 Welcome to My Project Portfolio!")
    st.sidebar.success("Select a page from the sidebar.")
    st.header('Created by Hardik Patil (Data Enthusiast)')
    st.markdown("Here, you'll find a collection of my data-driven projects, showcasing my skills in **machine learning**, **deep learning**, and **analytics**. From exploring patterns 📊 to **building AI models** 🤖, each project reflects my passion for turning data into insights.\n✨ Let’s dive into the world of data! 🚀")

    st.write("This is the main app page. Use the sidebar to navigate between pages.")


    ######################################### Profiles   #################################################


    # Function to encode image as Base64
    def get_base64_image(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()

    st.divider()
    st.text("Click to connect with me ...")
    
    # Convert images to Base64
    hacker_rank_img = get_base64_image("Images/icons/HackerRank.png")
    github_img = get_base64_image("Images/icons/Github.png")
    linkedin_img = get_base64_image("Images/icons/Linkedin.png")

    # Display social media icons
    st.html(f'''
    <p align="left" style="display: flex; gap: 50px;">
        <a href="https://www.hackerrank.com/profile/hardikmpatil23" target="_blank">
            <img src="data:image/png;base64,{hacker_rank_img}" alt="HackerRank" title="HackerRank Profile" height="40" width="40" />
        </a>
        <a href="https://github.com/Hardikpatil23-HP" target="_blank">
            <img src="data:image/png;base64,{github_img}" alt="GitHub" title="GitHub Profile" height="40" width="40" />
        </a>
        <a href="https://www.linkedin.com/in/hardik-patil-164066226" target="_blank">
            <img src="data:image/png;base64,{linkedin_img}" alt="LinkedIn" title="LinkedIn Profile" height="40" width="40" />
        </a>
    </p>
    ''')    
elif option == "INFO":

    st.sidebar.success("You have selected Project Info.")
    st.header("welcome to project info")
    option = st.radio("Select the model to view :", ["ANN", "CNN"],horizontal=True)
    st.divider()
    if option == "ANN":
        st.header("ANN Model")
        st.text("This model is used in Bike Resale Estimation project")
        st.header('Model Architecture')
        st.image('Images/Bike/ANN_arc.png')
        st.divider()
        st.header('Accuracy of Model')
        st.image('Images/Bike/Bike_Accuracy.png')
        st.divider()
        st.header('Mean Absolute Percentage Error')
        st.image('Images/Bike/Bike_Mape.png')
        st.divider()
        st.header('Feature_importance of Model')
        st.image('Images/Bike/Feature_importance.png')
    elif option == "CNN":
        st.header("CNN Model")
        st.text("This model is used in Cat and Dog Classification project")
        st.header('Model Architecture')
        st.image('Images/CatVsDog/CNN_arc.png')
        st.divider()
        st.header('F1_score of Model')
        st.image('Images/CatVsDog/f1_score.png')
        st.divider()
        st.header('Loss')
        st.image('Images/CatVsDog/Loss_cd.png')
        st.divider()


elif option == "ANN":

    st.sidebar.success("You have selected ANN project to view.")
    st.title("Welcome to Bike Resale Price Estimation Project")
    image = Image.open('Images/Bike/Bike2.png')
    colb1, colb2 = st.columns(2)
    with colb1:
        st.image(image,width=200)
    with colb2:
        st.title("Regression Model")
    st.divider()
    st.text("select specification below:")


    #########################################    brand  ##################################################

    brand = st.selectbox('brand :',  df.brand.unique())

    ########################################    bike_name  ##################################################

    selected_df = df[df['brand']==brand]

    Model = st.selectbox('Model :',  selected_df.cleaned_model.unique())

    ####################################    power  ##################################################

    Power_df = df[df['cleaned_model'] == Model]
    power_lvl = Power_df.power.sort_values().unique().astype(int)
    power = st.selectbox("Engine's Power :",  power_lvl)

    #########################################    city  ##################################################

    city  =  st.selectbox('City :',  df.city.sort_values().unique(),)

    ########################################    kms_driven  ##################################################

    owner = st.selectbox('Ownership :',df.owner.unique())
    
    #######################################    age  ##################################################

    age = st.number_input("Bike's Age :",min_value = 1,max_value = 20,step=1,format="%d")


    #######################################    age  ##################################################

    kms_input = st.number_input("Kms bike completed :",min_value = 0,max_value = 100000,step=100,format="%d")

    ################################### what is selected #####################################################

    st.divider()

    data = {'Specifications': ["Brand",
                               "Model",
                               "Power",
                               "City",
                               "Ownership",
                               "Age",
                               "Kms"
                               ], 
            'Values': [brand,
                       Model,
                       f"{power} cc",
                       city,
                       owner,
                       f"{int(age)} yr",
                       f"{int(kms_input)} Km"
                       ]
            }

    st.table(data)
    

    ########################################      Adjusting inputs    ################################################

    bike_input = int(df[(df['cleaned_model'] == Model) & (df['power'] == power)]['bike_id'].unique())
    city_input = int(df[df['city'] == city]['city_id'].unique())
    brand_input = int(df[df['brand'] == brand]['brand_id'].unique())
    ownership_input = int(df[df['owner'] == owner]['owner_level'].unique())

    input_data =    [  
                    kms_input ,
                    ownership_input, 
                    age, 
                    power, 
                    bike_input, 
                    city_input, 
                    brand_input
                    ]

    ######################################      Reshaping for Model       #########################################

    new_data = np.array(input_data).reshape(1, -1)  # Ensure shape (1,7)

    ######################################      loading Model       #########################################

    loaded_model = keras.models.load_model('Pages/Models/bikepredictor.keras')

    ######################################      Predicting Outcome       #########################################



    st.divider()
    if st.button('Calculate'):
        st.text("wait to load estimation...")
        prediction = loaded_model.predict(new_data)
        Answer = math.ceil(prediction / 100) * 100
        st.write('# The Resale price for above specification - Rs.', Answer)
    else:
        st.text("Click 'Calculate' to estimate price")
    

    st.text("The values here are predicted form a practice dataset may not be accurate.")


elif option == "CNN":

    st.title("Welcome to CAT and Dog Classification Project")
    st.sidebar.success("You have selected CNN project to view.")
    st.divider()
    image = Image.open('Images/CatVsDog/CatANDdog.png')
    coli, colt = st.columns(2)
    with coli:
        st.image(image,width=200)
    with colt:
        st.title("Classification Model")

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



