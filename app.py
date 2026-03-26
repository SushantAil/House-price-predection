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
    # create zero array
    x = [0] * len(data_columns)

    # assign numeric values
    try:
        x[data_columns.index('total_sqft')] = sqft
        x[data_columns.index('bath')] = bath
        x[data_columns.index('bhk')] = bhk
    except ValueError:
        st.error("Column names mismatch!")
        return None

    # assign location
    location = location.lower()
    if location in data_columns:
        loc_index = data_columns.index(location)
        x[loc_index] = 1

    # convert to dataframe with correct column order
    df = pd.DataFrame([x], columns=data_columns)

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
