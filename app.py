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
# Preprocessing Function (FIXED)
# -------------------------------
def preprocess_data(location, sqft, bath, bhk):
    # create dataframe with all columns = 0
    df = pd.DataFrame([0]*len(data_columns)).T
    df.columns = data_columns

    # fill numeric values
    df.at[0, 'total_sqft'] = sqft
    df.at[0, 'bath'] = bath
    df.at[0, 'bhk'] = bhk

    # handle location
    location = location.lower().strip()
    if location in data_columns:
        df.at[0, location] = 1

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
# Prediction (FIXED)
# -------------------------------
if st.button("Predict"):
    input_data = preprocess_data(location, sqft, bath, bhk)

    # FORCE correct column alignment (CRITICAL FIX)
    input_data = input_data.reindex(columns=data_columns, fill_value=0)

    prediction = model.predict(input_data)

    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")
