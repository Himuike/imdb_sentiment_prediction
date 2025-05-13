import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# --- Load Tokenizer and Model ---
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

model = tf.keras.models.load_model('cnn_model.h5')

# --- Constants ---
MAX_LEN = 200

# --- App Title ---
st.set_page_config(page_title="IMDB Sentiment Classifier", layout="centered")
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.markdown("Enter a movie review below to predict whether it's **Positive** or **Negative**.")

# --- User Input ---
review = st.text_area("📝 Your Review:", height=200)

if st.button("Predict Sentiment"):
    if not review.strip():
        st.warning("Please enter a review before predicting.")
    else:
        # Preprocess
        seq = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded)[0][0]
        sentiment = "Positive 😊" if prediction >= 0.5 else "Negative 😞"
        prob = round(prediction * 100, 2) if prediction >= 0.5 else round((1 - prediction) * 100, 2)

        # Result
        st.subheader("🎯 Sentiment Prediction:")
        if prediction >= 0.5:
            st.success(f"**{sentiment}** with {prob}% confidence")
        else:
            st.error(f"**{sentiment}** with {prob}% confidence")

st.markdown("---")
st.markdown("Made with ❤️ by Himanshu Uike")

