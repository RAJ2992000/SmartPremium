import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

st.title("Insurance Premium Predictor")
st.write("Enter customer details to predict premium amount.")

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
    
    # Needed for engineered columns
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
        
        # Required date breakdown
        "Policy Start Year": today.year,
        "Policy Start Month": today.month,
        "Policy Start Day": today.day,
        
        # Required
        "Policy Age (Days)": 0,
        "Days_Since_Policy_Start": 0,
        
        # Defaults (model required)
        "Customer_Feedback_Score": 1,
    }])

    return df

# -------- PREPROCESS (match model) --------
def preprocess(df):
    # Age Group
    df["Age Group"] = pd.cut(df["Age"], [18, 30, 45, 60, 100], labels=["18–30", "31–45", "46–60", "60+"])

    # Income Bracket
    df["Income_Bracket"] = pd.cut(df["Annual Income"], [0, 30000, 60000, 100000, np.inf],
                                  labels=["Low", "Median", "High", "Very High"])

    # Credit category
    df["Credit_Category"] = pd.cut(df["Credit Score"], [0, 400, 600, 800, np.inf], labels=False)

    # Dependents group
    df["Dependents_Group"] = df["Number of Dependents"].apply(
        lambda x: "None" if x == 0 else "Few" if x <= 2 else "Many"
    )

    # Age × Health
    df["Age_x_Health"] = df["Age"] * df["Health Score"]

    # Credit × Claims
    df["CreditScore_x_PrevClaims"] = df["Credit Score"] * df["Previous Claims"]

    # Flags
    df["Is_Smoker"] = df["Smoking Status"].apply(lambda x: 1 if x == "Yes" else 0)
    df["Low_Credit_Score"] = df["Credit Score"].apply(lambda x: 1 if x < 600 else 0)
    df["Multiple_Claims"] = df["Previous Claims"].apply(lambda x: 1 if x > 2 else 0)

    # Exercise score
    exercise_map = {"Daily": 4, "Weekly": 3, "Monthly": 2, "Rarely": 1, "Never": 0}
    df["Exercise_Freq_Score"] = df["Exercise Frequency"].map(exercise_map)

    # Income × Credit
    df["Income_x_Credit"] = df["Annual Income"] * df["Credit Score"]

    return df

# --------- PREDICT ---------
input_df = user_input()

if st.button("Estimate Premium"):
    final_df = preprocess(input_df)
    prediction = loaded_model.predict(final_df)[0]
    st.success(f"Estimated Premium: ₹{prediction:,.2f}")
