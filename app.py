import streamlit as st
import joblib

clf = joblib.load('/content/models/scaler.sav')

st.title('Klasifikasi Tingkat Obesitas')

# Input pengguna
height = st.number_input('Tinggi (cm)', min_value=0.5, max_value=2.5, value=1.75)
weight = st.number_input('Berat Badan (kg)', min_value=20, max_value=200, value=70)
age = st.number_input('Umur)', min_value=1, max_value=120, value=25)
gender = st.selectbox('Jenis kelamin', ('Female', 'Male'))

# Konversi gender ke bentuk numerik
gender_num = 0 if gender == 'Female' else 1

# Prediksi
input_data = np.array([[height, weight, age, gender_num]])
prediction = model.predict(input_data)

# Tampilkan hasil
obesity_level = ['Normal', 'Overweight', 'Obese']
st.write(f'The predicted obesity level is: {obesity_level[prediction[0]]}')
