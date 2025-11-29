import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================
#  Page Config
# ============================
st.set_page_config(
    page_title="Insurance Premium Predictor",
    layout="centered"
)

# ============================
#  BACKGROUND + ANIMATION CSS
# ============================
page_bg = """
<style>

[data-testid="stAppViewContainer"] {
    background-image: url('background.jpg');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
}

/* Make main container fade-in smoothly */
.main-container {
    animation: fadeIn 1.2s ease-in-out;
    background: rgba(255, 255, 255, 0.78);
    padding: 30px;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    backdrop-filter: blur(5px);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0px); }
}

/* Prediction text – BLACK */
.prediction-text {
    color: black !important;
    font-size: 28px;
    font-weight: 700;
}

</style>
"""

st.markdown(page_bg, unsafe_allow_html=True)

# Wrap app inside animated container
st.markdown("<div class='main-container'>", unsafe_allow_html=True)


# ============================
# Load model
# ============================
with open("best_modelup.pkl", "rb") as f:
    loaded_model = pickle.load(f)

st.markdown("<h1 style='text-align:center; color:#004aad;'>💰 Insurance Premium Predictor</h1>", unsafe_allow_html=True)
st.write("Enter customer details to predict premium amount.")


# ============================
# User Inputs
# ============================
def user_input():
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    annual_income = st.number_input("Annual Income", 0, 5000000, 50000)
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_dependents = st.number_input("Number of Dependents", 0, 10, 1)
    education = st.selectbox("Education Level", ["High School", "Graduate", "Post Graduate", "PhD", "Other"])
    occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Student", "Retired", "Other"])
    health_score = st.number_input("Health Score", 0, 100, 70)
    location = st.selectbox("Location", ["Urban", "Rural", "Semi-Urban"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Standard", "Premium", "Gold", "Platinum"])
    previous_claims = st.number_input("Previous Claims", 0, 20, 0)
    vehicle_age = st.number_input("Vehicle Age (Years)", 0, 20, 5)
    credit_score = st.number_input("Credit Score", 300, 900, 650)
    insurance_duration = st.number_input("Insurance Duration (Years)", 1, 30, 1)
    smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely", "Never"])
    property_type = st.selectbox("Property Type", ["Owned", "Rented", "Leased"])
    
    today = pd.Timestamp.today()
    
    df = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Annual Income": annual_income,
        "Marital Status": marital_status,
        "Number of Dependents": num_dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": health_score,
        "Location": location,
        "Policy Type": policy_type,
        "Previous Claims": previous_claims,
        "Vehicle Age": vehicle_age,
        "Credit Score": credit_score,
        "Insurance Duration": insurance_duration,
        "Smoking Status": smoking_status,
        "Exercise Frequency": exercise_frequency,
        "Property Type": property_type,

        "Policy Start Year": today.year,
        "Policy Start Month": today.month,
        "Policy Start Day": today.day,

        "Policy Age (Days)": 0,
        "Days_Since_Policy_Start": 0,

        "Customer_Feedback_Score": 1,
    }])

    return df


# ============================
# Preprocessing
# ============================
def preprocess(df):
    df["Age Group"] = pd.cut(df["Age"], [18, 30, 45, 60, 100], labels=["18–30", "31–45", "46–60", "60+"])
    df["Income_Bracket"] = pd.cut(df["Annual Income"], [0, 30000, 60000, 100000, np.inf],
                                  labels=["Low", "Median", "High", "Very High"])
    df["Credit_Category"] = pd.cut(df["Credit Score"], [0, 400, 600, 800, np.inf], labels=False)

    df["Dependents_Group"] = df["Number of Dependents"].apply(
        lambda x: "None" if x == 0 else "Few" if x <= 2 else "Many"
    )

    df["Age_x_Health"] = df["Age"] * df["Health Score"]
    df["CreditScore_x_PrevClaims"] = df["Credit Score"] * df["Previous Claims"]

    df["Is_Smoker"] = df["Smoking Status"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Low_Credit_Score"] = df["Credit Score"].apply(lambda x: 1 if x < 600 else 0)
    df["Multiple_Claims"] = df["Previous Claims"].apply(lambda x: 1 if x > 2 else 0)

    exercise_map = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Rarely": 1, "Never": 0}
    df["Exercise_Freq_Score"] = df["Exercise Frequency"].map(exercise_map)

    df["Income_x_Credit"] = df["Annual Income"] * df["Credit Score"]

    return df


# ============================
# Prediction
# ============================
input_df = user_input()

if st.button("Estimate Premium"):
    final_df = preprocess(input_df)
    prediction = loaded_model.predict(final_df)[0]

    st.markdown(
        f"<p class='prediction-text'>Estimated Premium: ₹{prediction:,.2f}</p>",
        unsafe_allow_html=True
    )

# Close main container
st.markdown("</div>", unsafe_allow_html=True)
