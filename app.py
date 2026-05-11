import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf # type: ignore

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

# ---------------- LOAD FILES WITH CACHE ----------------


@st.cache_resource
def load_all_files():

    model = tf.keras.models.load_model(
        "model.h5",
        compile=False
    )

    with open("gen_encoder.pkl", "rb") as file:
        gen_encoder = pickle.load(file)

    with open("geo_encoder.pkl", "rb") as file:
        geo_encoder = pickle.load(file)

    with open("scaler.pkl", "rb") as file:
        scaler = pickle.load(file)

    return model, gen_encoder, geo_encoder, scaler

# ---------------- LOADING SPINNER ----------------
try:
    with st.spinner("Loading model and files... Please wait ⏳"):
        model, gen_encoder, geo_encoder, scaler = load_all_files()

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

        # Create Input Dictionary
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

        # Convert to DataFrame
        data_df = pd.DataFrame([data])

        # Geography Encoding
        geo_encoded = geo_encoder.transform(
            pd.DataFrame([[data['Geography']]], columns=['Geography'])
        ).toarray()

        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=geo_encoder.get_feature_names_out(['Geography'])
        )

        # Drop Original Geography Column
        data_df.drop('Geography', axis=1, inplace=True)

        # Gender Encoding
        data_df['Gender'] = gen_encoder.transform(data_df['Gender'])

        # Merge Encoded Geography
        data_df = pd.concat([data_df, geo_encoded_df], axis=1)

        # Scale Data
        data_scaled = scaler.transform(data_df)

        # Prediction
        prediction = model.predict(data_scaled)

        prediction_proba = prediction[0][0]

        # ---------------- OUTPUT ----------------

        st.subheader("Prediction Result")

        if prediction_proba > 0.5:
            st.error(
                f"⚠️ Customer is likely to churn.\n\nProbability: {prediction_proba:.2f}"
            )
        else:
            st.success(
                f"✅ Customer is not likely to churn.\n\nProbability: {prediction_proba:.2f}"
            )

    except Exception as e:
        st.error(f"❌ Prediction Error: {e}")
