import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

np.random.seed(42)

@st.cache_resource
def train_model():
    n = 20000 
    
    solar_flux = np.random.normal(150, 50, n)
    kp_index = np.random.uniform(0, 9, n)
    wind_speed = np.random.normal(400, 150, n)
    bz = np.random.uniform(-20, 15, n)
    density = np.random.normal(5, 10, n)

    base_score = (kp_index * 40) + (wind_speed * 0.15) - (bz * 35) + (density * 5)
    noise = np.random.normal(0, 50, n) 
    
    total_score = base_score + noise

    storm = (total_score > 700).astype(int)

    X = np.column_stack([solar_flux, kp_index, wind_speed, bz, density])
    
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=4, 
        min_samples_leaf=50,
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
