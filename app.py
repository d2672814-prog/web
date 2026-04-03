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

flux = st.number_input("Күн белсенділігі", 50, 400, 150)
kp = st.number_input("Магниттік индекс", 0, 9, 4)
speed = st.number_input("Күн желінің жылдамдығы", 300, 1000, 400)
bz = st.number_input("Магнит өрісінің бағыты", -50, 20, 0)
density = st.number_input("Тығыздық", 0, 100, 5)

if st.button("Анализ жасау"):
    X = [[flux, kp, speed, bz, density]]
    prob = model.predict_proba(X)[0][1] * 100

    st.write(f"Risk: {prob:.1f}%")

    if prob >= 75:
        st.error("ӨТЕ ҚАУІПТІ")
    elif prob >= 40:
        st.warning("ҚАУІПТІ")
    else:
        st.success("ТҰРАҚТЫ")
