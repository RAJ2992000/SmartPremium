import streamlit as st
import pandas as pd
import pickle

# -------------------------------
# SIMPLE FAST BACKGROUND
# -------------------------------
def set_bg(url):
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("{url}");
            background-size: cover;
            background-position: center;
        }}
        </style>
    """, unsafe_allow_html=True)

# Lightweight background (small file, fast to load)
set_bg("https://images.unsplash.com/photo-1504610926078-a1611febcad3?auto=format&fit=crop&w=1400&q=60")

# -------------------------------
# LOAD MODEL
# -------------------------------
with open("best_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

# -------------------------------
# CENTERED BIG TITLE
# -------------------------------
st.markdown(
    """
    <h1 style="
        text-align: center;
        font-size: 60px;
        font-weight: 900;
        color: white;
        margin-top: 10px;
        text-shadow: 3px 3px 10px rgba(0,0,0,0.7);
    ">
        Insurance Premium Predictor 💰
    </h1>
    """,
    unsafe_allow_html=True
)

st.sidebar.write("Enter customer details to predict premium amount.")

# -------------------------------
# USER INPUT (Everything in LEFT SIDEBAR)
# -------------------------------
def user_input():
    age = st.sidebar.number_input("Age", 18, 100, 35)
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    Annual_Income = st.sidebar.number_input("Annual Income", 0, 50000)
    marital_status = st.sidebar.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    Number_of_Dependents = st.sidebar.number_input("Number of Dependents", 0, 5, 1)
    education = st.sidebar.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.sidebar.selectbox("Occupation", ["Salaried", "Self-Employed", "Retired"])
    Health_Score = st.sidebar.slider("Health Score (0–100)", 0.0, 100.0, 50.0)
    Location = st.sidebar.selectbox("Location", ["Urban", "Suburban", "Rural"])
    Previous_Claims = st.sidebar.number_input("Previous Claims", 0, 10, 0)
    Vehicle_Age = st.sidebar.number_input("Vehicle Age (years)", 0, 20, 5)
    Credit_Score = st.sidebar.number_input("Credit Score", 300, 900, 650)
    Insurance_Duration = st.sidebar.number_input("Insurance Duration (years)", 0, 10, 3)
    Policy_Start_Date = st.sidebar.date_input("Policy Start Date")
    Customer_Feedback = st.sidebar.selectbox("Customer Feedback", ["Poor", "Average", "Good", "Excellent"])
    Smoking_Status = st.sidebar.selectbox("Smoking Status", ["Yes", "No"])
    Exercise_Frequency = st.sidebar.selectbox("Exercise Frequency", ["None", "Rarely", "Occasional", "Monthly", "Weekly", "Daily"])
    Property_Type = st.sidebar.selectbox("Property Type", ["House", "Apartment", "Condo"])
    policy_type = st.sidebar.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])

    policy_start_date = pd.to_datetime(Policy_Start_Date)
    policy_age_days = (pd.Timestamp.today() - policy_start_date).days

    data = {
        "Age": age,
        "Gender": gender,
        "Annual Income": Annual_Income,
        "Marital Status": marital_status,
        "Number of Dependents": Number_of_Dependents,
        "Education Level": education,
        "Occupation": occupation,
        "Health Score": Health_Score,
        "Location": Location,
        "Previous Claims": Previous_Claims,
        "Vehicle Age": Vehicle_Age,
        "Credit Score": Credit_Score,
        "Insurance Duration": Insurance_Duration,
        "Policy Start Date": policy_start_date,
        "Customer Feedback": Customer_Feedback,
        "Smoking Status": Smoking_Status,
        "Exercise Frequency": Exercise_Frequency,
        "Property Type": Property_Type,
        "Policy Type": policy_type,
        "Policy Age (Days)": policy_age_days,
        "Policy Start Year": policy_start_date.year,
        "Policy Start Month": policy_start_date.month,
        "Policy Start Day": policy_start_date.day
    }

    return pd.DataFrame([data])

input_df = user_input()

# -------------------------------
# YOUR SAME PREPROCESSING FUNCTION
# -------------------------------
def preprocess_input(df):
    import numpy as np
    df = df.copy()

    for col in df.select_dtypes(include=['float64','int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'])

    df['Policy Age (Days)'] = (pd.Timestamp("today") - df['Policy Start Date']).dt.days
    df['Policy Start Year'] = df['Policy Start Date'].dt.year
    df['Policy Start Month'] = df['Policy Start Date'].dt.month
    df['Policy Start Day'] = df['Policy Start Date'].dt.day

    df['Age Group'] = pd.cut(df['Age'], [18,30,45,60,100],
                             labels=['18–30','31–45','46–60','60+'])

    df['Income_Bracket'] = pd.cut(df['Annual Income'],
                                  [0,30000,60000,100000,np.inf],
                                  labels=['Low','Median','High','Very High'])

    df['Credit_Category'] = pd.cut(df['Credit Score'], [0,400,600,800,np.inf],
                                   labels=False)

    df['dependents group'] = df['Number of Dependents'].apply(
        lambda x:'None' if x==0 else 'Few' if x<=2 else 'Many')

    df['Days_Since_Policy_Start'] = (pd.Timestamp.now() - df['Policy Start Date']).dt.days

    df['Customer_Feedback_Score'] = df['Customer Feedback'].map({
        'Poor':0, 'Average':1, 'Good':2
    })

    df['Age_x_Health'] = df['Age'] * df['Health Score']
    df['CreditScore_x_PrevClaims'] = df['Credit Score'] * df['Previous Claims']
    df['Is_Smoker'] = df['Smoking Status'].apply(lambda x:1 if x.lower()=="yes" else 0)
    df['Low_Credit_Score'] = df['Credit Score'].apply(lambda x:1 if x<600 else 0)
    df['Multiple_Claims'] = df['Previous Claims'].apply(lambda x:1 if x>2 else 0)

    df['Exercise_Freq_Score'] = df['Exercise Frequency'].map(
        {'Daily':4,'Weekly':3,'Monthly':2,'Rarely':1,'None':0})

    df['Income_x_Credit'] = df['Annual Income'] * df['Credit Score']

    for col in df.select_dtypes(include=['object','category']).columns:
        df[col] = df[col].astype(str)

    return df

# -------------------------------
# BUTTON + CENTER PREDICTION
# -------------------------------
if st.sidebar.button("Estimate Premium"):
    processed = preprocess_input(input_df)
    prediction = loaded_model.predict(processed)[0]

    # 🎈 Balloons animation
    st.balloons()

    # Center output using HTML
    st.markdown(
        f"""
        <div style="
            display: flex;
            justify-content: center;
            margin-top: 40px;
        ">
            <div style="
                background: rgba(255,255,255,0.85);
                padding: 30px 50px;
                border-radius: 15px;
                border-left: 10px solid #ff9900;
                box-shadow: 0px 5px 15px rgba(0,0,0,0.4);
                text-align: center;
                width: 60%;
            ">
                <h1 style="color:#ff5500; font-size:50px; margin-bottom:10px;">
                    ₹{prediction:,.2f}
                </h1>
                <h3 style="color:black; font-size:28px; font-weight:600;">
                    Estimated Premium
                </h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

