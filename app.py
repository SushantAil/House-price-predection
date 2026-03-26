import streamlit as st
import pandas as pd
import joblib
import json

# -------------------------------
# Load Model and Columns
# -------------------------------
model = joblib.load('banglore_home_prices_model.pickle')

with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']


# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(location, sqft, bath, bhk):
    x = [0] * len(data_columns)

    # numeric features
    x[0] = sqft
    x[1] = bath
    x[2] = bhk

    # one-hot encoding for location
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    return pd.DataFrame([x], columns=data_columns)


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🏠 House Price Prediction")

st.subheader("Enter House Details")

# Inputs
location = st.selectbox("Select Location", data_columns[3:])
sqft = st.number_input("Total Square Feet", min_value=300)
bath = st.number_input("Number of Bathrooms", min_value=1)
bhk = st.number_input("BHK", min_value=1)


# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):
    input_data = preprocess_data(location, sqft, bath, bhk)
    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")
