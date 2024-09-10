import streamlit as st
import pickle
import numpy as np
import json

# Load the model and columns
with open('bangalore_home_prices_model.pickle', 'rb') as f:
    model = pickle.load(f)

with open('columns.json', 'r') as f:
    data_columns = json.load(f)['data_columns']

# Extracting the locations (assuming location columns are all the ones after the first three features)
locations = [col for col in data_columns[3:]]

# Define the prediction function
def predict_price(location, sqft, bath, bhk):
    # Initialize the input array with zeros
    x = np.zeros(len(data_columns))
    
    # Assign input values
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    
    # If location is found in the encoded columns, set its value to 1
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1
    
    # Make the prediction using the loaded model
    return model.predict([x])[0]

# Streamlit UI
st.title('Bangalore Home Price Prediction')

# Get user input
location = st.selectbox('Location', locations)
sqft = st.number_input('Total Square Feet', min_value=500)
bath = st.number_input('Number of Bathrooms', min_value=1, max_value=10)
bhk = st.number_input('BHK', min_value=1, max_value=10)

# When button is pressed, predict the price
if st.button('Predict Price'):
    if sqft > 0 and bath > 0 and bhk > 0:
        price = predict_price(location, sqft, bath, bhk)
        st.success(f"The predicted price is ₹ {price:.2f} Lakhs")
    else:
        st.error("Please enter valid inputs")
