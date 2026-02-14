import streamlit as st

st.write("APP STARTED") 
import streamlit as st
import pandas as pd
import pickle
import numpy as np

# --- 1. Muat Model dan Scaler yang Telah Disimpan ---

try:
    with open('best_model_gradient_boosting.pkl', 'rb') as model_file:
        loaded_model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        loaded_scaler = pickle.load(scaler_file)
    st.success("Model dan Scaler berhasil dimuat!")
except FileNotFoundError:
    st.error("Error: Pastikan file 'best_model_gradient_boosting.pkl' dan 'scaler.pkl' ada di direktori yang sama.")
    st.stop()
except Exception as e:
    st.error(f"Terjadi kesalahan saat memuat file: {e}")
    st.stop()

# --- 2. Definisikan Pemetaan dan Urutan Kolom ---

pendidikan_mapping = {"D3": 0, "S1": 1, "SMA": 2, "SMK": 3}
jurusan_mapping = {"Administrasi": 0, "Desain Grafis": 1, "Otomotif": 2, "Teknik Las": 3, "Teknik Listrik": 4}

expected_columns_order = [
    'Usia', 'Durasi_Jam', 'Nilai_Ujian', 'Pendidikan', 'Jurusan',
    'Jenis_Kelamin_Laki-laki', 'Jenis_Kelamin_Wanita',
    'Status_Bekerja_Belum Bekerja', 'Status_Bekerja_Sudah Bekerja'
]

# --- 3. Antarmuka Streamlit ---

st.set_page_config(page_title="Prediksi Gaji Pertama", layout="centered")
st.title("💰 Prediksi Gaji Pertama Lulusan Pelatihan Vokasi")
st.write("Aplikasi ini memprediksi gaji pertama (dalam jutaan Rupiah) berdasarkan profil peserta pelatihan.")
st.markdown("--- # Ini adalah contoh saja dan bukan nilai prediksi actual, jika ingin nilai actual silahkan jalankan aplikasi streamlit Anda sendiri # ---")

st.header("Masukkan Informasi Peserta")

# Input dari Pengguna
usia = st.number_input("Usia (Tahun)", min_value=18, max_value=60, value=25, step=1)
durasi_jam = st.number_input("Durasi Pelatihan (Jam)", min_value=20, max_value=100, value=60, step=1)
nilai_ujian = st.number_input("Nilai Ujian (Skala 0-100)", min_value=0.0, max_value=100.0, value=80.0, step=0.1)
pendidikan = st.selectbox("Pendidikan Terakhir", list(pendidikan_mapping.keys()), index=list(pendidikan_mapping.keys()).index('SMK'))
jurusan = st.selectbox("Jurusan Pelatihan", list(jurusan_mapping.keys()), index=list(jurusan_mapping.keys()).index('Desain Grafis'))
jenis_kelamin = st.radio("Jenis Kelamin", ['Laki-laki', 'Wanita'], index=1)
status_bekerja = st.radio("Status Bekerja Saat Ini", ['Sudah Bekerja', 'Belum Bekerja'], index=0)

st.markdown("--- # Ini adalah contoh saja dan bukan nilai prediksi actual, jika ingin nilai actual silahkan jalankan aplikasi streamlit Anda sendiri # ---")

if st.button("Prediksi Gaji"):
    # Buat dictionary dari input pengguna
    user_input_raw = {
        'Usia': usia,
        'Durasi_Jam': durasi_jam,
        'Nilai_Ujian': nilai_ujian,
        'Pendidikan': pendidikan,
        'Jurusan': jurusan,
        'Jenis_Kelamin': jenis_kelamin,
        'Status_Bekerja': status_bekerja
    }

    input_df = pd.DataFrame([user_input_raw])

    # Preprocessing (sesuai dengan notebook sebelumnya)
    input_df['Pendidikan'] = input_df['Pendidikan'].map(pendidikan_mapping)
    input_df['Jurusan'] = input_df['Jurusan'].map(jurusan_mapping)

    input_df['Jenis_Kelamin_Laki-laki'] = (input_df['Jenis_Kelamin'] == 'Laki-laki').astype(int)
    input_df['Jenis_Kelamin_Wanita'] = (input_df['Jenis_Kelamin'] == 'Wanita').astype(int)
    input_df['Status_Bekerja_Belum Bekerja'] = (input_df['Status_Bekerja'] == 'Belum Bekerja').astype(int)
    input_df['Status_Bekerja_Sudah Bekerja'] = (input_df['Status_Bekerja'] == 'Sudah Bekerja').astype(int)

    input_df = input_df.drop(columns=['Jenis_Kelamin', 'Status_Bekerja'])

    # Reindex untuk memastikan urutan kolom yang benar
    input_df = input_df[expected_columns_order]

    # Scaling fitur
    input_df_scaled_array = loaded_scaler.transform(input_df)
    input_df_scaled = pd.DataFrame(input_df_scaled_array, columns=expected_columns_order)

    # Prediksi
    predicted_salary = loaded_model.predict(input_df_scaled)

    st.subheader('Hasil Prediksi Gaji Awal')
    st.success(f"### Estimasi Gaji Pertama: Rp {predicted_salary[0] * 1_000_000:,.2f}")
    st.write("*(Prediksi dalam jutaan Rupiah)*")
    st.info("Catatan: Prediksi ini adalah estimasi berdasarkan model yang dilatih. Hasil sebenarnya dapat bervariasi.")
