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
    x = [0] * len(data_columns)

    # numeric features
    x[data_columns.index('total_sqft')] = float(sqft)
    x[data_columns.index('bath')] = float(bath)
    x[data_columns.index('bhk')] = float(bhk)

    # location
    location = location.lower().strip()
    if location in data_columns:
        x[data_columns.index(location)] = 1

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
# Prediction (FIXED)
# -------------------------------
if st.button("Predict"):
    input_data = preprocess_data(location, sqft, bath, bhk)

    try:
        prediction = model.predict(input_data)
    except:
        # fallback if sklearn complains
        prediction = model.predict(input_data.values)

    st.success(f"Estimated Price: ₹ {round(prediction[0], 2)} Lakhs")
