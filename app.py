import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

@st.cache_resource
def train_model():
    n = 15000 
    
    # Параметры
    solar_flux = np.random.normal(150, 50, n)
    kp_index = np.random.uniform(0, 9, n)
    wind_speed = np.random.normal(400, 150, n)
    bz = np.random.uniform(-20, 10, n)
    density = np.random.normal(5, 10, n)

    score = (kp_index * 20) + (wind_speed * 0.05) - (bz * 15) + (density * 2)
    
    score += np.random.normal(0, 15, n)

    storm = (score > 200).astype(int)

    X = np.column_stack([solar_flux, kp_index, wind_speed, bz, density])
   
    model = RandomForestClassifier(
        n_estimators=150, 
        max_depth=5, 
        min_samples_leaf=20,
        random_state=42
    )
    model.fit(X, storm)
    return model
model = train_model()

st.title("🚀 SolarShield AI")

flux = st.number_input("Solar Flux", 50, 400, 150)
kp = st.number_input("Magnit Index", 0, 9, 4)
speed = st.number_input("Solar Wind Speed", 300, 1000, 400)
bz = st.number_input("Bz", -50, 20, 0)
density = st.number_input("Density", 0, 100, 5)

if st.button("Analyz"):
    X = [[flux, kp, speed, bz, density]]
    prob = model.predict_proba(X)[0][1] * 100

    st.write(f"Risk: {prob:.1f}%")

    if prob >= 75:
        st.error("Criticial")
    elif prob >= 40:
        st.warning("Warning")
    else:
        st.success("Stable")
