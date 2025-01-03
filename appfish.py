import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Memuat model dan scaler
model = joblib.load('svm_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl')  # Pastikan label encoder disimpan jika belum

# Judul Aplikasi
st.title("Prediksi Spesies Ikan menggunakan SVM")

# Input dari pengguna
st.header("Masukkan Data Ikan")
length = st.slider("Panjang Ikan (cm)", min_value=0.0, step=0.1)
weight = st.slider("Berat Ikan (kg)", min_value=0.0, step=0.1)
w_l_ratio = st.slider("Rasio Berat/Panjang", min_value=0.0, step=0.01)

if st.button('Prediksi'):
    # Menyiapkan data input
    input_data = np.array([[length, weight, w_l_ratio]])

    # Lakukan scaling pada data input
    input_scaled = scaler.transform(input_data)

    # Prediksi spesies
    prediction = model.predict(input_scaled)
    predicted_species = label_encoder.inverse_transform(prediction)

    # Tampilkan hasil prediksi
    st.success(f"Spesies ikan yang diprediksi: {predicted_species[0]}")
