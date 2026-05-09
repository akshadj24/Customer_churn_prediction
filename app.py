import streamlit as st
import pickle
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}

h1 {
    color: #204570;
    text-align: center;
}

.stButton > button {
    background-color: #204570;
    color: white;
    font-size: 18px;
    border-radius: 10px;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #2f6bb2;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.title("📊 Customer Churn Prediction")

# ---------------- LOAD MODEL & FILES ----------------
try:
    model = load_model("model.h5")
    
    with open("gen_encoder.pkl", "rb") as file:
        gen_encoder = pickle.load(file)

    with open("geo_encoder.pkl", "rb") as file:
        geo_encoder = pickle.load(file)

    # Make sure your file name is scaler.pkl
    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    st.success("✅ Model Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error Loading Files: {e}")
    st.stop()

# ---------------- INPUT FIELDS ----------------

CreditScore = st.slider(
    label="Credit Score",
    min_value=0,
    max_value=850,
    value=500
)

Geography = st.selectbox(
    label="Country",
    options=["France", "Germany", "Spain"]
)

Gender = st.selectbox(
    label="Gender",
    options=["Male", "Female"]
)

Age = st.number_input(
    label="Age",
    min_value=18,
    max_value=100,
    value=25
)

Tenure = st.slider(
    label="Tenure",
    min_value=1,
    max_value=10,
    value=5
)

Balance = st.number_input(
    label="Balance",
    min_value=0.0,
    value=1000.0
)

NumOfProducts = st.slider(
    label="Number Of Products",
    min_value=1,
    max_value=10,
    value=1
)

HasCrCard = st.selectbox(
    label="Has Credit Card",
    options=[0, 1]
)

IsActiveMember = st.selectbox(
    label="Is Active Member",
    options=[0, 1]
)

EstimatedSalary = st.number_input(
    label="Estimated Salary",
    min_value=0.0,
    value=50000.0
)

# ---------------- PREDICTION BUTTON ----------------

if st.button("Predict Churn"):

    try:

        # Create DataFrame
        data = {
            "CreditScore": CreditScore,
            "Geography": Geography,
            "Gender": Gender,
            "Age": Age,
            "Tenure": Tenure,
            "Balance": Balance,
            "NumOfProducts": NumOfProducts,
            "HasCrCard": HasCrCard,
            "IsActiveMember": IsActiveMember,
            "EstimatedSalary": EstimatedSalary
        }

        data_df = pd.DataFrame([data])

        # Encode Geography
        geo_encoded = geo_encoder.transform(
            pd.DataFrame([[data['Geography']]], columns=['Geography'])
        ).toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=geo_encoder.get_feature_names_out(['Geography'])
        )

        # Drop original Geography column
        data_df.drop('Geography', axis=1, inplace=True)

        # Encode Gender
        data_df['Gender'] = gen_encoder.transform(data_df['Gender'])

        # Combine encoded geography
        data_df = pd.concat([data_df, geo_encoded_df], axis=1)

        # Scale data
        data_scaled = scaler.transform(data_df)

        # Prediction
        prediction = model.predict(data_scaled)

        prediction_proba = prediction[0][0]

        st.subheader("Prediction Result")

        if prediction_proba > 0.5:
            st.error(
                f"⚠️ The customer is likely to churn.\n\nProbability: {prediction_proba:.2f}"
            )
        else:
            st.success(
                f"✅ The customer is not likely to churn.\n\nProbability: {prediction_proba:.2f}"
            )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
