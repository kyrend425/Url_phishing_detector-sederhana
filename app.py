import streamlit as st
import numpy as np
import re
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# --- Load Pretrained Model, Tokenizer, and Scaler ---
# Ganti 'model.h5', 'tokenizer.pkl', dan 'scaler.pkl' dengan file yang sesuai jika model/tokenizer/scaler Anda sudah disimpan.

model = load_model("model.h5")

# Memuat tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Memuat scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.write("Tokenizer dan Scaler berhasil dimuat!")

# --- Feature Extraction Functions ---
def special_char_ratio(url):
    return len(re.findall(r"[^a-zA-Z0-9]", url)) / len(url) if len(url) > 0 else 0

def preprocess_single_url(url):
    # Preprocessing for a single URL
    url_sequence = tokenizer.texts_to_sequences([url])  # Tokenize the URL
    padded_sequence = pad_sequences(url_sequence, maxlen=100, padding='post', truncating='post')
    
    # Feature engineering
    url_length = len(url)
    special_char_ratio_val = special_char_ratio(url)
    numerical_features = [[url_length, special_char_ratio_val]]
    scaled_numerical_features = scaler.transform(numerical_features)  # Scale features
    
    return padded_sequence, scaled_numerical_features

def predict_url(url):
    padded_sequence, scaled_features = preprocess_single_url(url)
    prediction = model.predict([padded_sequence, scaled_features])
    return "Phishing" if prediction[0][0] > 0.5 else "Bukan Phishing"

# --- Streamlit App ---
st.title("URL Phishing Detector")
st.write("Masukkan URL untuk memprediksi apakah URL tersebut phishing atau bukan.")

# Input field
url_input = st.text_input("Masukkan URL", "")

# Predict button
if st.button("Prediksi"):
    if url_input.strip() == "":
        st.warning("Silakan masukkan URL yang valid.")
    else:
        result = predict_url(url_input)
        st.success(f"Prediksi: {result}")
