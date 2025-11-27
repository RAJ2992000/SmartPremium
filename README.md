# 💰 SmartPremium: Predicting Insurance Costs with Machine Learning

## 📌 Project Overview  
SmartPremium is a complete end-to-end Machine Learning project designed to predict **insurance premium amounts** using customer demographics, health metrics, policy details, and claim history.  
The project covers everything from **EDA → Preprocessing → ML Modeling → MLflow Tracking → Streamlit Deployment**.

This project replicates a real industry workflow used by insurance and finance companies.

---

## 🎯 Skills Learned  
- Data Preprocessing & Cleaning  
- Exploratory Data Analysis (EDA)  
- Feature Engineering  
- Regression Model Development  
- Hyperparameter Tuning  
- ML Pipeline Creation  
- MLflow Experiment Tracking  
- Streamlit Web App Deployment  
- Git/GitHub Version Control  

---

## 🧩 Problem Statement  
Insurance companies estimate the premium for each customer using multiple risk-based features.  
Your goal is to build a **predictive machine learning model** that estimates the *Premium Amount* based on these inputs.

---

## 💼 Business Use Cases  
- **Premium Optimization:** Insurance firms can price policies based on risk.  
- **Loan Risk Assessment:** Banks can estimate customer liability.  
- **Healthcare Forecasting:** Providers can anticipate medical cost trends.  
- **Customer Support:** Generate instant premium quotes for new customers.  

---

## 🛠️ Project Workflow

### 📌 Step 1 — Understanding & Exploring the Dataset  
- Load dataset and examine structure  
- Identify missing values and incorrect data types  
- Analyze distributions (age, income, claims, etc.)  
- Check correlations with target (Premium Amount)  
- Visualize relationships (histograms, pair-plots, heatmaps)

---

### 📌 Step 2 — Data Preprocessing  
- Handle missing values (median/mode)  
- Encode categorical variables (Label/OneHot Encoding)  
- Convert date columns  
- Feature scaling for numeric columns  
- Train-test split (80%-20%)

---

### 📌 Step 3 — Model Development  
Regression models used:
- **Linear Regression**  
- **Decision Tree Regressor**  
- **Random Forest Regressor**  
- **XGBoost Regressor**  

Evaluation Metrics:
- RMSE  
- MAE  
- R² Score  
- RMSLE  

The model with the best accuracy is saved for deployment.

---

### 📌 Step 4 — ML Pipeline + MLflow  
- Build end-to-end ML Pipeline  
- Log models, metrics, and parameters using MLflow  
- Compare experiments and store best model  

---

### 📌 Step 5 — Streamlit Deployment  
A clean and interactive Streamlit app allows users to input:
- Age  
- Income  
- Health Score  
- Policy Type  
- Claims History  
- Location  
…and more.

The app outputs a **real-time predicted insurance premium** using the trained model.

---

## 📊 Dataset Overview  
- 200,000+ records  
- 20+ features  
- Mix of numerical, categorical, date, and text variables  
- Includes:  
  - Age  
  - Annual Income  
  - Marital Status  
  - Education  
  - Occupation  
  - Health Score  
  - Policy Type  
  - Previous Claims  
  - Vehicle Age  
  - Credit Score  
  - Smoking / Exercise habits  
  - Property Type  
  - Policy Start Date  
  - Customer Feedback  
- Target: **Premium Amount**

Data contains:
- Missing values  
- Outliers  
- Incorrect data types  
- Skewed numeric distributions  
(simulating real-world insurance datasets)

---

## 📁 Project Deliverables  
Your final submission includes:
- ✔ Jupyter Notebook with full workflow  
- ✔ ML Pipeline + MLflow integration  
- ✔ Trained ML model (.pkl)  
- ✔ Streamlit app code  
- ✔ Documentation and results  

---

## 🧰 Tech Stack  
- Python  
- Pandas, NumPy  
- Scikit-Learn  
- XGBoost  
- Matplotlib, Seaborn  
- MLflow  
- Streamlit  
- Git/GitHub  

---

## 🧪 Evaluation Metrics  
- 📉 Root Mean Squared Error (RMSE)  
- 📉 Mean Absolute Error (MAE)  
- 📈 R² Score  
- 📉 Root Mean Squared Log Error (RMSLE)

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
