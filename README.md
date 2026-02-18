# 🌍 Air Quality Index (AQI) Prediction System  
### Supervised Machine Learning Major Project

<p align="center">
  <img src="https://img.shields.io/badge/Machine%20Learning-Supervised-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Status-Completed-success?style=for-the-badge"/>
</p>

---

## 📌 Project Overview

The **Air Quality Index (AQI) Prediction System** is an end-to-end **supervised machine learning project** designed to predict air pollution levels based on environmental and pollutant parameters.  

This project covers the **complete ML lifecycle**:
- Data understanding
- Data cleaning & preprocessing
- Outlier detection
- Model training & evaluation
- Real-time prediction using a web interface

The system also provides **air quality levels and health impact interpretation**, making it suitable for real-world environmental monitoring applications.

---

## 🎯 Problem Statement

> To predict the **Air Quality Index (AQI)** using supervised machine learning algorithms based on pollutant concentration and environmental data, and to present the results through an interactive web application.

---

## 📂 Dataset Description

**Dataset Name:** `air_quality_health_impact_data.csv`

### Key Features:
- PM2.5
- PM10
- NO₂
- SO₂
- O₃
- CO (if available)
- Temperature
- Humidity
- Wind Speed
- Other environmental indicators

### Target Variable:
- **AQI (Air Quality Index)** — numerical value representing air pollution severity.

📌 The dataset was cleaned, preprocessed, and used for both **regression and classification tasks**.

---

## 🧪 AQI Interpretation

| AQI Range | Air Quality Level | Health Impact |
|----------|------------------|---------------|
| 0 – 50 | Good | Air is clean, no health risk |
| 51 – 100 | Satisfactory | Minor discomfort to sensitive people |
| 101 – 200 | Moderate | Breathing discomfort for lung patients |
| 201 – 300 | Poor | Breathing discomfort for most people |
| 301 – 400 | Very Poor | Respiratory illness on prolonged exposure |
| 401 – 500 | Severe | Serious health impact |

---

## 📓 Jupyter Notebook Work (Data Science Phase)

📁 **Folder:** `notebooks/`

### Implemented Steps:
- Data loading and exploration
- Handling missing values
- Removing duplicate records
- Outlier detection using statistical methods
- Feature selection
- Feature scaling using StandardScaler
- Train-test splitting
- Training multiple supervised ML models
- Model evaluation and comparison

### Algorithms Used:
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Logistic Regression (AQI Category Classification)

📌 This notebook demonstrates **core machine learning knowledge and data preprocessing expertise**.

---

## 🌐 VS Code + Streamlit Work (Application Phase)

📁 **Folder:** `app/`

### Web Application Features:
- Light, professional UI
- Sidebar-based user input
- Model selection dropdown
- Real-time AQI prediction
- Air quality level display
- Health impact explanation
- Clean and user-friendly design

📌 This phase showcases **deployment skills and real-world application development**.

---

## 📊 Output & Results

- Accurate AQI prediction using multiple ML models
- Best-performing models identified using evaluation metrics
- Visual and textual interpretation of results
- Interactive prediction system for end users

📸 *(Screenshots of the Streamlit UI can be added in `/screenshots` folder)*

---

## 🧠 Machine Learning Models Summary

| Model | Task | Evaluation Metric |
|-----|------|------------------|
| Linear Regression | AQI Prediction | R² Score |
| Decision Tree | AQI Prediction | R² Score |
| Random Forest | AQI Prediction | R² Score |
| Logistic Regression | AQI Classification | Accuracy |

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Matplotlib & Seaborn
- Jupyter Notebook
- VS Code

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
streamlit run app/app.py

🎓 Academic Relevance

Designed and implemented as a Major Project in accordance with undergraduate engineering curriculum standards.

Demonstrates the complete supervised machine learning workflow, including data preprocessing, model training, evaluation, and deployment.

Aligned with the academic requirements of the B.E. (Artificial Intelligence) program.

Suitable for project review, viva voce examinations, and campus placement evaluations.

👩‍💻 Author

Geethanjali M
Bachelor of Engineering — Artificial Intelligence

🌱 Conclusion

This project effectively illustrates the application of supervised machine learning techniques to a real-world environmental problem. By integrating data preprocessing, predictive modeling, and web-based deployment, the system delivers accurate Air Quality Index (AQI) predictions along with meaningful health impact interpretations. The project highlights both technical competence and practical problem-solving skills, making it a valuable academic and professional contribution.

