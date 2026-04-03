import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

@st.cache_resource
def train_model():
    n = 1500
    solar_flux = np.random.normal(150, 50, n)
    magnetic = np.random.normal(4, 2, n)
    wind_speed = np.random.normal(400, 120, n)
    bz = np.random.uniform(-15, 15, n)
    density = np.random.normal(5, 3, n)

    score = (wind_speed * 0.4) + (density * 20) - (bz * 30)
    storm = (score > 600).astype(int)

    X = np.column_stack([solar_flux, magnetic, wind_speed, bz, density])

    model = RandomForestClassifier(n_estimators=100, max_depth=8)
    model.fit(X, storm)
    return model

model = train_model()

st.title("🚀 SolarShield AI")

flux = st.number_input("Solar Flux", 50, 300, 150)
kp = st.number_input("Magnetic Index (Kp)", 0, 9, 4)
speed = st.number_input("Solar Wind Speed", 300, 800, 400)
bz = st.number_input("Bz", -20, 20, 0)
density = st.number_input("Density", 0, 20, 5)

if st.button("Анализ жасау"):
    X = [[flux, kp, speed, bz, density]]
    prob = model.predict_proba(X)[0][1] * 100

    st.write(f"Risk: {prob:.1f}%")

    if prob >= 75:
        st.error("CRITICAL")
    elif prob >= 40:
        st.warning("WARNING")
    else:
        st.success("STABLE")
