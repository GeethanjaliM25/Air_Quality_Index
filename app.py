import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Quality Prediction System",
    page_icon="🌍",
    layout="wide"
)

# ---------------- LIGHT THEME CSS ----------------
st.markdown("""
<style>
body {
    background-color: #f8fafc;
}
.main {
    background-color: #f8fafc;
}
h1, h2, h3 {
    text-align: center;
    color: #0f172a;
}
.sidebar .sidebar-content {
    background-color: #e2e8f0;
}
.result-box {
    background-color: #e0f2fe;
    padding: 30px;
    border-radius: 18px;
    text-align: center;
    border: 2px solid #38bdf8;
}
.result-box h1, .result-box h2, .result-box h3, .result-box p {
    color: #000000;
    font-weight: 600;
}
.footer {
    text-align: center;
    color: #334155;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown("<h1>🌍 Air Quality Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<h3>Major Project | Supervised Machine Learning</h3>", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_csv("air_quality_health_impact_data.csv")
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)
    return df

df = load_data()

# ---------------- FEATURES & TARGET ----------------
X = df.drop("AQI", axis=1)
y = df["AQI"]

y_class = pd.cut(
    y,
    bins=[0, 50, 100, 200, 300, 500],
    labels=[0, 1, 2, 3, 4]
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Model Configuration")

model_choice = st.sidebar.selectbox(
    "Select ML Model",
    ("Linear Regression", "Decision Tree", "Random Forest", "Logistic Regression")
)

test_size = st.sidebar.slider("Test Size (%)", 10, 40, 20) / 100

st.sidebar.markdown("### 🌫️ Input Parameters")

user_inputs = []
for feature in X.columns:
    val = st.sidebar.slider(
        feature,
        float(df[feature].min()),
        float(df[feature].max()),
        float(df[feature].mean())
    )
    user_inputs.append(val)

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=test_size, random_state=42
)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X_scaled, y_class, test_size=test_size, random_state=42
)

# ---------------- TRAIN MODEL ----------------
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X_train, y_train)
    metric = f"R² Score: {r2_score(y_test, model.predict(X_test)):.3f}"

elif model_choice == "Decision Tree":
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    metric = f"R² Score: {r2_score(y_test, model.predict(X_test)):.3f}"

elif model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    metric = f"R² Score: {r2_score(y_test, model.predict(X_test)):.3f}"

else:
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_c, y_train_c)
    metric = f"Accuracy: {accuracy_score(y_test_c, model.predict(X_test_c)):.3f}"

st.markdown(f"<h3 style='text-align:center;'>📊 {metric}</h3>", unsafe_allow_html=True)

# ---------------- AQI INTERPRETATION ----------------
def aqi_interpretation(aqi):
    if aqi <= 50:
        return "Good", "Air is clean and poses no health risk."
    elif aqi <= 100:
        return "Satisfactory", "Minor discomfort to sensitive people."
    elif aqi <= 200:
        return "Moderate", "Breathing discomfort for people with lung disease."
    elif aqi <= 300:
        return "Poor", "Breathing discomfort for most people."
    elif aqi <= 400:
        return "Very Poor", "Respiratory illness on prolonged exposure."
    else:
        return "Severe", "Serious health impact."

# ---------------- CENTER PREDICTION ----------------
st.markdown("## 🧪 Prediction Result")

if st.button("🚀 Predict AQI"):
    user_scaled = scaler.transform(np.array(user_inputs).reshape(1, -1))

    if model_choice == "Logistic Regression":
        category = model.predict(user_scaled)[0]
        st.markdown(f"""
        <div class="result-box">
            <h2>Predicted AQI Category</h2>
            <h1>{category}</h1>
            <p>Represents the severity level of air pollution.</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        aqi = model.predict(user_scaled)[0]
        level, meaning = aqi_interpretation(aqi)

        st.markdown(f"""
        <div class="result-box">
            <h2>Predicted AQI Value</h2>
            <h1>{aqi:.2f}</h1>
            <h3>Air Quality Level: {level}</h3>
            <p>{meaning}</p>
        </div>
        """, unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<div class='footer'>🌱 Air Quality Prediction | Supervised ML Major Project</div>",
    unsafe_allow_html=True
)
