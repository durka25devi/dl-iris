import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

# -------------------------------
# 1. Load Model and Preprocessors
# -------------------------------
model = load_model("iris_ann_model.h5")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classification using ANN")
st.write("This app predicts the **species of an Iris flower** based on its sepal and petal measurements.")

# -------------------------------
# 3. Input Fields
# -------------------------------
st.subheader("Enter Flower Measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# -------------------------------
# 4. Prediction Logic
# -------------------------------
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)
    species_name = le.inverse_transform([predicted_class])[0]

    st.success(f"🌼 Predicted Iris Species: **{species_name}**")

    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame(prediction, columns=le.classes_)
    st.dataframe(prob_df.style.highlight_max(axis=1))

# -------------------------------
# 5. Footer
# -------------------------------
st.markdown("""
---
🌸 **Created with Streamlit & TensorFlow**  
By AI Enthusiast 🤖
""")
