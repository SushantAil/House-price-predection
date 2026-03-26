import streamlit as st
import pandas as pd
import joblib
import json

# -------------------------------
# Load Model and Columns
# -------------------------------
model = joblib.load('model.pkl')

with open("columns.json", "r") as f:
    data_columns = json.load(f)['data_columns']


# -------------------------------
# Preprocessing Function
# -------------------------------
def preprocess_data(location, sqft, bath, bhk):
    # create empty dataframe with all columns
    df = pd.DataFrame(columns=data_columns)
    df.loc[0] = 0

    # fill numeric values (IMPORTANT names)
    df.at[0, 'total_sqft'] = sqft
    df.at[0, 'bath'] = bath
    df.at[0, 'bhk'] = bhk

    # set location column = 1
    if location.lower() in data_columns:
        df.at[0, location.lower()] = 1

    return df


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
