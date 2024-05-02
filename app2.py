import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd

# Sayfa Ayarları
st.set_page_config(
    page_title="Breast Cancer Detection App",
    page_icon="logo.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Modeli yükle
model = load_model('my_model.h5', compile=False)

# Streamlit arayüzünü oluştur

st.markdown("<h1 style='color: #FF1493; text-align: center;'>🔬🧠💻Breast Cancer Detection💻🧠🔬</h1>", unsafe_allow_html=True)
st.image("background.jpg", use_column_width=True)

st.markdown("<div style='text-align: center;'><p style='color: #435c76; font-size: 22px;'>Breast cancer is a type of cancer that forms in the cells of the breast. It can occur in both men and women, but it is far more common in women. Breast cancer usually begins in the milk-producing ducts (invasive ductal carcinoma) or the lobules (invasive lobular carcinoma) of the breast. However, it can also begin in other areas of the breast.</p></div>", unsafe_allow_html=True)

data_dict= {
    "Prevalence": "Breast cancer is one of the most common cancers worldwide, particularly among women. It is estimated that about 1 in 8 women will develop breast cancer during their lifetime. Additionally, though less common, men can also develop breast cancer",
    "Impact on Health:": "Breast cancer can have a significant impact on a person's health and quality of life. If left untreated or undetected, it can spread to other parts of the body, leading to serious complications and potentially death.",
    "Early Detection":" Detecting breast cancer early greatly improves the chances of successful treatment and survival. Regular screenings, such as mammograms, can help detect breast cancer in its early stages when it is most treatable." }

data_df = pd.DataFrame(data_dict.items(), columns=["Feature", "Description"])
st.markdown("<h3 style='color: #435c76;'>Breast cancer is important for several reasons:</h3>", unsafe_allow_html=True)
st.table(data_df.style.set_properties(**{'background-color': '#FF1493', 'color': "white"}))



st.markdown("<div style='text-align: center;'><p style='color: #435c76; font-size: 22px;'>The main goal of the project was to classify breast lesions as benign and malignant, thus enabling early diagnosis and correct treatment. In this project, I aimed to increase the clinical applicability and automation of the model by achieving high levels of precision and accuracy. In addition, I paid great attention to the issues of explainability and reliability by strengthening the generalizability of the model. These results are only the insights of the deep learning algorithm. It does not contain a certainty, consult your doctor for precise information.</p></div>", unsafe_allow_html=True)




# Resim yükleme
st.sidebar.markdown("<span style='color: #435c76; font-size: 22px;'>User Input Data 📥</span>", unsafe_allow_html=True)



uploaded_file = st.sidebar.file_uploader(" ", type=["jpg", "jpeg", "png"])



if uploaded_file is not None:
    # Yüklenen resmi göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Resmi modelin girdi formatına dönüştür ve normalleştir
    image_array = np.array(image.resize((150, 150))) / 255.0  # Örnek boyut: 150x150

    # Tahmin yap
    prediction = model.predict(np.expand_dims(image_array, axis=0))

    # Tahmin sonucunu görüntüle
    st.sidebar.markdown("<h2 style='color: #435c76 ;'>Cell Cluster Status Result 📍</h2>", unsafe_allow_html=True)
    if prediction > 0.5:
        st.sidebar.write("Predict: **Malignant**")
    else:
        st.sidebar.write("Predict: **Benign**")


st.sidebar.markdown("<h2 style='color: #435c76; text-align: center;'>For Contact</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>Developer: Busra SULUKAN</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>E-mail: bsulukan18@gmail.com</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center;'>LinkedIn : https://www.linkedin.com/in/b%C3%BC%C5%9Fra-sulukan-82299a177/</p>", unsafe_allow_html=True)


