import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

# -------------------------------
# 1. Page Configuration
# -------------------------------
st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")

st.title("🌸 Iris Flower Classification using ANN")
st.write("This app predicts the **species of an Iris flower** based on its sepal and petal measurements.")

# -------------------------------
# 2. Load and Prepare Data
# -------------------------------
df = pd.read_csv("iris.csv")

# Features and target
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = df['Species']

# Encode target labels
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 3. Build Model
# -------------------------------
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),
    Dense(8, activation='relu'),
    Dense(3, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train model
model.fit(X_scaled, y, epochs=50, batch_size=4, verbose=0)

# -------------------------------
# 4. User Input Section
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
# 5. Prediction
# -------------------------------
if st.button("🔍 Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_class = np.argmax(prediction)
    species_name = le.inverse_transform([predicted_class])[0]

    st.success(f"🌼 Predicted Iris Species: **{species_name}**")

    st.write("---")
    st.write("### Prediction Probabilities")
    prob_df = pd.DataFrame(prediction, columns=le.classes_)
    st.dataframe(prob_df.style.highlight_max(axis=1))

# -------------------------------
# 6. Footer
# -------------------------------
st.markdown("""
---
🌸 **Created with Streamlit & TensorFlow**  
By AI Enthusiast 🤖
""")
