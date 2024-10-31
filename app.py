# Import necessary libraries
import streamlit as st
import joblib
import numpy as np

# Define the CropRecommendationModel class
class CropRecommendationModel:
    def __init__(self, model, crop_names):
        self.model = model
        self.crop_names = crop_names
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        predicted_crops = []
        for row in y_pred:
            index = np.argmax(row)
            predicted_crops.append(self.crop_names[index] if row[index] == 1 else "None")
        return predicted_crops

# Load the model
    from joblib import load
    model =  joblib.load("crop_recommendation_model.joblib")

# Sidebar for page navigation
page = st.sidebar.selectbox("Select a page", ["Home", "About Us", "Important Information"])

# Home page for crop recommendation
if page == "Home":
    #st.title("AI-Powered Crop Recommendation Model")
    st.markdown("<h1 style='text-align: center;'>AI-Powered Crop Recommendation Model</h1>", unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    /* Main page background color */
    .stApp {
        background-color: #A8E6A1; /* Light green */
    }
    
    /* Sidebar background color */
    .css-1d391kg { 
        background-color: #BDFCC9; /* Mint green for sidebar */
    }
    </style>
    """,
    unsafe_allow_html=True
)

    st.sidebar.markdown("""
## Project Description

The AI-Powered Crop Recommendation Model uses Advance Machine Learning Technique to provide farmers with customized crop suggestions based on soil and environmental factors. 

Inspired by the need to help farmers optimize yield and manage resources, particularly in climate-challenged areas, this project aims to boost agricultural productivity and promote sustainable practices. It emphasizes the importance of technology in connecting traditional farming with modern techniques, ultimately enhancing food security and economic stability for farming communities.
""")
    # Input fields for the seven features in a compact layout
    col1, col2, col3 = st.columns(3)

    with col1:
        N = st.number_input("Nitrogen Content (N)", min_value=0, max_value=171)
        temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=44.0)
        ph = st.number_input("pH Level", min_value=4.0, max_value=8.0)

    with col2:
        P = st.number_input("Phosphorous Content (P)", min_value=5, max_value=112)
        humidity = st.number_input("Humidity (%)", min_value=30.0, max_value=100.0)

    with col3:
        K = st.number_input("Potassium Content (K)", min_value=5, max_value=107)
        rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=1708.0)
    # Input fields for the seven features
    #N = st.number_input("Nitrogen Content (N)", min_value=0, max_value=171)
    #P = st.number_input("Phosphorous Content (P)", min_value=5, max_value=112)
    #K = st.number_input("Potassium Content (K)", min_value=5, max_value=107)
    #temperature = st.number_input("Temperature (°C)", min_value=10.0, max_value=44.0)
   # humidity = st.number_input("Humidity (%)", min_value=30.0, max_value=100.0)
   # ph = st.number_input("pH Level", min_value=4.0, max_value=8.0)
    #rainfall = st.number_input("Rainfall (mm)", min_value=20.0, max_value=1708.0)

    # Submit button to make predictions
    if st.button("Predict"):
        # Prepare input features for the model
        input_features = [[N, P, K, temperature, humidity, ph, rainfall]]
        prediction = model.predict(input_features)
        
        # Display the result
        st.write(f"Recommended Crop:             {prediction[0]}")
       


    # Like button
    if st.button("Like ❤️"):
        st.success("Thank you for liking the model!")

    # Additional information at the bottom of the home page
    st.write("""
    ---
    **Important Information**  
    Please ensure that you provide accurate inputs for nitrogen, phosphorous, potassium, temperature, humidity, pH, and rainfall.  
    This model is designed for general recommendations and may not capture specific local conditions.  
    Consult with local agronomists for additional guidance.
    """)



# About Us page
elif page == "About Us":
    st.title("About Us")
    st.write("""
    Welcome to the AI-Powered Crop Recommendation Model. 
    This tool is designed to help farmers make informed decisions on crop selection based on soil and climate conditions.
    Our mission is to support sustainable agriculture through advanced technology.

    
    ## MEET THE TEAM
    
    ### AYOOLA MUJIB AYODELE
    
    FE/23/89361170

    COHORT 2

    Learning Track : AI and ML
    
    Agricultural AI/ML Specialist and Consultant
    """)

# Important Information page
elif page == "Important Information":
    st.title("Important Information")
    st.write("""
    Please ensure that you provide accurate inputs for nitrogen, phosphorous, potassium, temperature, humidity, pH, and rainfall.
    This model is designed for general recommendations and may not capture specific local conditions.
    
    Consult with local agronomists for additional guidance.
    """)
