#core pksgs
import streamlit as st
import os
import joblib
from PIL import Image
import matplotlib.pyplot as plt


#EDA pkgs

import pandas as pd
import numpy as np

def load_data(dataset):
    df=pd.read_csv(dataset)
    return df


### load model
    
def load_model_prediction(model_file):
    loaded_model=joblib.load(open(os.path.join(model_file),"rb"))
    return loaded_model
    
### creating diconery method for values
gender_label= {"Male": 0, "Female": 1, "Other":2}
work_type_label= {'Private': 0, 'Self-employed': 1, 'Govt_job':2, 'children':3, 'Never_worked':4}
Residence_type_label={'Urban': 0, 'Rural': 1}
smoking_status_label={'formerly smoked': 0, 'never smoked': 1, 'smokes':2, 'Unknown':3}
ever_married_label={'No': 0, 'Yes': 1} 
hypertension_label={'No': 0, 'Yes': 1}
heart_disease_label={'No': 0, 'Yes': 1}
strok_label={0:"TIDAK ada risiko untuk terkena penyakit stroke", 1:"RISIKO terkena penyakit stroke"}

data=load_data('stroke.csv')

age_min=data.age.min()
age_max=data.age.max()


avg_glucose_level_min=data.avg_glucose_level.min()
avg_glucose_level_max=data.avg_glucose_level.max()

bmi_min=data.bmi.min()
bmi_max=data.bmi.max()

## get keys

def get_value(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return value
## find the key from the dictonary
            
def get_keys(val,my_dict):
    for key,value in my_dict.items():
        if val==key:
            return key

def main():
      
    menu=["Project Data","Classifications"]
    choices=st.sidebar.selectbox("Menu",menu)


    if choices=="Project Data":
        st.title("Analisis Prediksi Stroke")
        st.caption("Ini adalah aplikasi yang bertujuan untuk menganalisis dan memprediksi risiko stroke berdasarkan beberapa faktor yang diberikan.")
        st.markdown("---")
        st.subheader("Petunjuk Penggunaan")
        st.markdown("""
        Pada bagian **Data Proyek**, Anda dapat melihat informasi tentang proyek dan contoh data yang digunakan.
        Jika Anda ingin melihat ringkasan metrik dari data, Anda dapat mencentang opsi "Tampilkan Ringkasan Metrik".

        Silakan pilih menu yang Anda inginkan di sebelah kiri.
        - **Data Proyek**: Menampilkan informasi tentang proyek dan contoh data yang digunakan.
        - **Klasifikasi**: Memungkinkan Anda untuk memasukkan data dan melakukan prediksi risiko stroke.

        Pada bagian **Klasifikasi**, Anda dapat memasukkan nilai-nilai faktor-faktor yang diperlukan untuk melakukan prediksi risiko stroke.
        Setelah memasukkan nilai-nilai tersebut, klik tombol "Evaluate" untuk melihat hasil prediksi.
        """)
        
        data=load_data('stroke.csv')
        st.dataframe(data.head(10))
        
        
        if st.checkbox("Tampilkan Ringkasan Metrik"):
            st.write(data.describe())

        st.markdown("---")        
        st.markdown("""Kami berharap aplikasi ini dapat memberikan manfaat bagi Anda dalam memahami dan mengidentifikasi risiko stroke. Dengan menggunakan aplikasi ini, diharapkan Anda dapat lebih menyadari pentingnya faktor-faktor yang berkontribusi terhadap risiko stroke dan mengambil langkah-langkah pencegahan yang tepat.""")
        st.markdown("---")        
        st.caption("""        
        Proyek ini dibuat oleh anggota tim kami yang terdiri dari:
        - Shopi Nurhidayanti (2006061)
        - Dini Antika (2006184)
        - Anisa Syifa Syafaat (2006072)
        - Dewi Sulastri (2006198)
        - Muhammad Ilyasa (2006175)      
        """)

    if choices == "Classifications":
        st.title("Analisis Klasifikasi Stroke")
        st.subheader("Klasifikasi")
        st.markdown("""
        Pada bagian ini, Anda dapat melakukan klasifikasi risiko stroke berdasarkan faktor-faktor yang diberikan.
        Silakan pilih nilai untuk setiap faktor pada panel sebelah kiri, kemudian klik tombol "Evaluate" untuk melihat hasil klasifikasi.

        Faktor-faktor yang dapat dipilih antara lain:
        - Gender (Jenis Kelamin)
        - Age (Usia)
        - Hypertension (Hipertensi)
        - Heart Disease (Penyakit Jantung)
        - Ever Married (Status Pernikahan)
        - Work Type (Jenis Pekerjaan)
        - Residence Type (Tipe Tempat Tinggal)
        - Average Glucose Level (Tingkat Rata-rata Glukosa)
        - BMI (Indeks Massa Tubuh)
        - Smoking Status (Status Merokok)

        Setelah memilih nilai-nilai yang diinginkan, hasil klasifikasi akan ditampilkan di bawah tombol "Evaluate".

        **Catatan:** Aplikasi ini menggunakan model klasifikasi C4.5 yang dioptimasi dengan Adaboost untuk memprediksi risiko stroke.
        """)
       
        gender=st.selectbox("Select the Gender",tuple(gender_label.keys()))
        age=st.number_input("Select the  Age of a person",age_min,age_max)
        hypertension=st.selectbox("Select the hypertension",tuple(hypertension_label.keys()))
        heart_disease=st.selectbox("Select the heart_disease",tuple(heart_disease_label.keys()))
        ever_married=st.selectbox("Select the ever_married",tuple(ever_married_label.keys()))
        work_type=st.selectbox("Select the work_type",tuple(work_type_label.keys()))
        Residence_type=st.selectbox("Select the Residence_type",tuple(Residence_type_label.keys()))
        avg_glucose_level=st.number_input("Select the avg_glucose_level",avg_glucose_level_min,avg_glucose_level_max)
        bmi=st.number_input("Select the bmi_level",bmi_min,bmi_max)
        smoking_status=st.selectbox("Select the smoking_status",tuple(smoking_status_label.keys()))
        
        ### encoding
        
        gender_v= get_value(gender,gender_label)
        hypertension_v= get_value(hypertension,hypertension_label)
        heart_disease_v= get_value(heart_disease,heart_disease_label)
        ever_married_v= get_value(ever_married,ever_married_label)
        work_type_v= get_value(work_type,work_type_label)
        Residence_type_v= get_value(Residence_type,Residence_type_label)
        smoking_status_v= get_value(smoking_status,smoking_status_label)
        
        
        pretty_data={
                "gender":gender,
                "age":age,
                "hypertension":hypertension,
                "heart_disease":heart_disease,
                "ever_married":ever_married,
                "work_type":work_type,
                "Residence_type":Residence_type,
                "avg_glucose_level":avg_glucose_level,
                "bmi":bmi,
                "smoking_status":smoking_status
                }
        st.subheader("Options Selected")
        st.json(pretty_data)
        
        st.subheader("Encoded data")
        encoding_data=[gender_v,age,hypertension_v,heart_disease_v,
                       ever_married_v,work_type_v,Residence_type_v,
                       avg_glucose_level,bmi,smoking_status_v]
        st.write(encoding_data)
        
        # input data should be 2d not one 1d so we convert here
        prep_encoding_data=np.array(encoding_data).reshape(1,-1)
        
        model_choice= ("Model_Choice C4.5 + Adaboost")
        if st.button("Evaluate"):
            predictor = load_model_prediction("models/adaboost_c45.pkl")
            prediction = predictor.predict(prep_encoding_data)
            final_result = get_value(prediction, strok_label)
            st.success(final_result)
            st.markdown("""
            **Catatan:**
            Hasil evaluasi yang ditampilkan di atas hanyalah prediksi berdasarkan model klasifikasi yang digunakan.
            Namun, penting untuk diingat bahwa hasil ini masih belum terlalu akurat dan tidak menggantikan diagnosis medis yang sebenarnya.
            Jika Anda memiliki kekhawatiran mengenai risiko stroke, disarankan untuk berkonsultasi dengan dokter atau tenaga medis yang kompeten.
            """) 

        st.markdown("---")        
        st.markdown("""Kami berharap aplikasi ini dapat memberikan manfaat bagi Anda dalam memahami dan mengidentifikasi risiko stroke. Dengan menggunakan aplikasi ini, diharapkan Anda dapat lebih menyadari pentingnya faktor-faktor yang berkontribusi terhadap risiko stroke dan mengambil langkah-langkah pencegahan yang tepat.""")
        st.markdown("---")        
        st.caption("""        
        Proyek ini dibuat oleh anggota tim kami yang terdiri dari:
        - Shopi Nurhidayanti (2006061)
        - Dini Antika (2006184)
        - Anisa Syifa Syafaat (2006072)
        - Dewi Sulastri (2006198)
        - Muhammad Ilyasa (2006175)      
        """)

if __name__=='__main__':
    main()
    
    