import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="SmartPremium Predictor", layout="wide")

# ============================================
# BACKGROUND IMAGE CSS
# ============================================
def add_bg_image():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url("https://images.unsplash.com/photo-1517816743773-6e0fd518b4a6");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .css-1d391kg, .css-12oz5g7 {
            background: rgba(0, 0, 0, 0.55) !important;
            backdrop-filter: blur(4px);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_image()

# ============================================
# Load Model
# ============================================
@st.cache_resource
def load_model():
    with open("best_model_new.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ============================================
# HEADER WITH PREMIUM FONT + SHADOW
# ============================================
st.markdown(
    """
    <h1 style='text-align:center; 
               color:#ffffff; 
               font-weight:900; 
               text-shadow: 3px 3px 10px #000;'>
        💰 SmartPremium Insurance Predictor
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h3 style='text-align:center; 
               color:#ffdd57; 
               text-shadow: 2px 2px 6px black; 
               margin-top:-10px;'>
        Fill details on the left panel and get the premium instantly!
    </h3>
    """,
    unsafe_allow_html=True
)

# ============================================
# SIDEBAR INPUT PANEL
# ============================================
st.sidebar.header("🔍 Enter Customer & Policy Details")
st.sidebar.markdown("---")

# ----------- Sidebar Inputs --------------
age = st.sidebar.number_input("Age", 18, 100, 30)
gender = st.sidebar.selectbox("Gender", ["Male", "Female", "Other"])
annual_income = st.sidebar.number_input("Annual Income", 1000, 5000000, 50000)
education = st.sidebar.selectbox("Education Level", ["High School", "Graduate", "Post Graduate", "PhD", "Other"])
occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Student", "Retired", "Other"])

marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
num_dependents = st.sidebar.number_input("Number of Dependents", 0, 15, 1)
health_score = st.sidebar.number_input("Health Score", 0, 100, 70)
location = st.sidebar.selectbox("Location", ["Urban", "Rural", "Semi-Urban"])
policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Standard", "Premium", "Gold", "Platinum"])

previous_claims = st.sidebar.number_input("Previous Claims", 0, 20, 0)
vehicle_age = st.sidebar.number_input("Vehicle Age (Years)", 0, 20, 5)
credit_score = st.sidebar.number_input("Credit Score", 300, 850, 650)
insurance_duration = st.sidebar.number_input("Insurance Duration (Years)", 1, 30, 1)
smoking_status = st.sidebar.selectbox("Smoking Status", ["No", "Yes"])

exercise_frequency = st.sidebar.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely", "Never"])
property_type = st.sidebar.selectbox("Property Type", ["Owned", "Rented", "Leased"])

predict_button = st.sidebar.button("💡 Predict Premium", use_container_width=True)

# ============================================
# BUILD INPUT DATAFRAME
# ============================================
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
}])

# ============================================
# REQUIRED DATE COLUMNS (FAST FIX)
# ============================================
today = pd.Timestamp.today()

df["Policy Start Year"] = today.year
df["Policy Start Month"] = today.month
df["Policy Start Day"] = today.day
df["Policy Age (Days)"] = 0
df["Days_Since_Policy_Start"] = 0

# ============================================
# ENGINEERED FEATURES
# ============================================
df["Age Group"] = pd.cut([age], [18, 30, 45, 60, 100], labels=["18–30", "31–45", "46–60", "60+"])
df["Customer_Feedback_Score"] = 1
df["Income_Bracket"] = pd.cut([annual_income], [0, 30000, 60000, 100000, np.inf], labels=["Low", "Median", "High", "Very High"])
df["Credit_Category"] = pd.cut([credit_score], [0, 400, 600, 800, np.inf], labels=False)
df["Dependents_Group"] = ["None" if num_dependents == 0 else "Few" if num_dependents <= 2 else "Many"]
df["Age_x_Health"] = age * health_score
df["CreditScore_x_PrevClaims"] = credit_score * previous_claims
df["Is_Smoker"] = 1 if smoking_status == "Yes" else 0
df["Low_Credit_Score"] = 1 if credit_score < 600 else 0
df["Multiple_Claims"] = 1 if previous_claims > 2 else 0

exercise_map = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Rarely": 1, "Never": 0}
df["Exercise_Freq_Score"] = exercise_map[exercise_frequency]

df["Income_x_Credit"] = annual_income * credit_score

# ============================================
# MAIN OUTPUT
# ============================================
if predict_button:

    pred = model.predict(df)[0]
    pred = round(pred, 2)

    st.balloons()

    st.markdown(
        st.markdown(
    f"""
    <div style='text-align:center; margin-top:40px;'>
        <h2 style='color:#ffdd57; text-shadow:2px 2px 6px black;'>💰 Estimated Premium Amount</h2>
        <h1 style='font-size:60px; color:#000000; font-weight:900; text-shadow:2px 2px 6px #ffffff;'>₹ {pred}</h1>
    </div>
    """,
    unsafe_allow_html=True
)
    )
