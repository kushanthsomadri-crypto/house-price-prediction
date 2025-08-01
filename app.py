import streamlit as st
import numpy as np
import xgboost as xgb

# Load trained model
model = xgb.XGBRegressor()
model.load_model("xgb_model.json")

# Streamlit page config
st.set_page_config(page_title="House Price Prediction")

st.title("🏠 House Price Prediction")
st.write("Enter the details below to predict house price:")

# Input fields
MedInc = st.number_input("Median Income")
HouseAge = st.number_input("House Age")
AveRooms = st.number_input("Average Number of Rooms")
AveBedrms = st.number_input("Average Number of Bedrooms")
Population = st.number_input("Population")
AveOccup = st.number_input("Average Occupancy")
Latitude = st.number_input("Latitude")
Longitude = st.number_input("Longitude")

# Prediction button
if st.button("Predict"):
    # Prepare input
    data = np.array([[MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]])
    
    # Predict
    result = model.predict(data)
    
    # Display result
    st.success(f"🏷️ Estimated House Price: ${result[0]*100000:.2f}")
