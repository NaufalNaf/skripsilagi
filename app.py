import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

st.write("Loading model...")
clf = joblib.load('models/scaler.sav')
st.write("Model loaded.")

st.title('Klasifikasi Tingkat Obesitas')

# Input pengguna
height = st.number_input('Tinggi (m)', min_value=0.5, max_value=2.5, value=1.75)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=200, value=70)
gender = st.selectbox('Jenis kelamin', ('Female', 'Male'))
st.write("Inputs received.")

# Konversi gender ke bentuk numerik
gender_num = 0 if gender == 'Female' else 1

# Prediksi
if st.button('Predict'):
    st.write("Button clicked.")

    # Buat array input
    input_data = np.array([[height, weight, gender_num]])

    # Debug: Tampilkan data input
    st.write(f'Input data: {input_data}')

    try:
        input_data_scaled = scaler.transform(input_data)
        prediction = model.predict(input_data_scaled)
        # Lakukan prediksi
        prediction = clf.predict(input_data)
        # Debug: Tampilkan prediksi mentah
        st.write(f'Raw prediction: {prediction}')
        
        # Tampilkan hasil prediksi
        obesity_level = ['Extremely Weak','weak','Normal', 'Overweight', 'Obese','Extremely Obese']
        st.write(f'The predicted obesity level is: {obesity_level[prediction[0]]}')
    except Exception as e:
        st.write(f'Error during prediction: {e}')
