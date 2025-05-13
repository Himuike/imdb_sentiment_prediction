import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf


# Load the trained model
model = tf.keras.models.load_model("cnn_model.h5")
# Load the tokenizer (saved previously during training)
import pickle
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Set padding length (same as in training phase)
maxlen = 100  # Adjust to your maxlen value

# Function to predict sentiment based on CNN model
def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(seq, maxlen=maxlen)
    pred = model.predict(pad)
    sentiment = "Positive" if pred[0][0] > 0.5 else "Negative"
    return sentiment

# Streamlit UI
st.title('Movie Review Sentiment Analysis')
st.write("Enter a movie review to predict its sentiment:")

# Text input for user review
user_review = st.text_area("Enter your movie review:")

if user_review:
    # Predict sentiment based on the user input review
    sentiment = predict_sentiment(user_review)
    
    st.write(f"**Predicted Sentiment:** {sentiment}")
