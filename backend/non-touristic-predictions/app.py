'''to run the application:
cd backend/non-touristic-predictions
streamlit run app.py
'''

import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model and scaler
rf_model = joblib.load("rf_model_5.joblib")  # Ensure this file exists
scaler = joblib.load("scaler_5_features.joblib")  # Ensure this file exists

top_features = [
    "price", "number_of_reviews_ltm", "number_of_reviews",
    "host_acceptance_rate", "host_total_listings_count"
]


# Function to get user input and make a prediction
def predict_success(user_input):
    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Normalize input using the trained scaler
    normalized_input = scaler.transform(input_df)

    # Get prediction probabilities
    probabilities = rf_model.predict_proba(normalized_input)[0]

    # Define class labels
    class_labels = ["Unsuccessful", "Moderate Success", "Very Successful"]

    # Select the class with the highest probability
    best_class_index = np.argmax(probabilities)
    prediction_label = class_labels[best_class_index]

    return prediction_label, probabilities




# Streamlit UI
st.title("🏠 Apartment Success Prediction")

st.write("Enter apartment details below to predict its success.")

# User input fields
user_input = {}
for feature in top_features:
    user_input[feature] = st.number_input(f"Enter {feature}", min_value=0, step=1, value=50)

# Predict button
if st.button("Predict Success"):
    prediction_label, probabilities = predict_success(user_input)

    # Display the prediction
    st.subheader(f"🔮 Prediction: {prediction_label}")

    # Show probabilities
    st.write("🔢 **Success Probability Breakdown:**")
    st.write(f"- Unsuccessful: {probabilities[0]:.2f}")
    st.write(f"- Moderate Success: {probabilities[1]:.2f}")
    st.write(f"- Very Successful: {probabilities[2]:.2f}")