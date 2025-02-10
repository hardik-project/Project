import streamlit as st
from PIL import Image


##########################################      Title      ##############################################

image = Image.open('Images/Bike/Bike2.png')
st.image(image,width=200)
st.title("Bike Resale Price Estimator APP")

st.text("Please select specifications")


######################################      Importing libs      ###########################################

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras
import math
import numpy as np


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

#########################################    brand  ##################################################

brand = st.selectbox('brand :',  df.brand.unique())

########################################    bike_name  ##################################################

selected_df = df[df['brand']==brand]

Model = st.selectbox('Model :',  selected_df.bike_name.unique())

####################################    power  ##################################################

Power_df = df[df['bike_name'] == Model]
power_lvl = Power_df.power.unique().astype(int)
power_lvl.sort()
power = st.selectbox("Model's Power :",  power_lvl)

#########################################    city  ##################################################

city  =  st.selectbox('City :',  df.city.unique(),)

########################################    kms_driven  ##################################################

owner = st.selectbox('Ownership :',df.owner.unique())

#######################################    age  ##################################################

age = st.number_input("Bike's Age :",min_value = df.age.min(),max_value = df.age.max(),format="%1f")


#######################################    age  ##################################################

kms_input = st.number_input("Kms bike completed :",min_value = df.kms_driven.min(),max_value = df.kms_driven.max(),format="%1f")

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

bike_input = int(df[df['bike_name'] == Model]['bike_id'].unique())
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

prediction = loaded_model.predict(new_data)
Answer = math.ceil(prediction / 100) * 100

st.divider()

st.write('# The Resale price for above specification : Rs.', Answer)

st.text("The values here are predicted form a practice dataset may not be accurate.")

