# 🎬 IMDb Sentiment Prediction

A complete end-to-end Natural Language Processing (NLP) project that predicts sentiment (Positive or Negative) from IMDb movie reviews using multiple Deep Learning models. The best-performing model is deployed using Streamlit Cloud.

## 🚀 Live Demo

🔗 [Try the App on Streamlit](https://imdbprediction.streamlit.app)  

---

## 📌 About the Project

This project involves analyzing IMDb movie reviews to determine whether the sentiment expressed is **positive or negative**. It combines traditional NLP preprocessing with various Deep Learning architectures such as:

- Artificial Neural Network (ANN)
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Bidirectional LSTM (BiLSTM)

The CNN model performed best and was deployed for real-time inference using Streamlit.

---

## 📊 Model Performance

| Model   | Test Accuracy (%) |
|---------|-------------------|
| ANN     | 87.20             |
| CNN ✅  | **86.65**         |
| RNN     | 85.06             |
| LSTM    | 85.46             |
| BiLSTM  | 85.46             |

✅ CNN model was chosen for deployment based on a trade-off between accuracy and inference time.

---

## ⚙️ Tech Stack

- **Language:** Python
- **Libraries:** TensorFlow, Keras, NLTK, NumPy, Pandas, Matplotlib
- **Model Deployment:** Streamlit Cloud
- **Dataset:** IMDb Movie Review Dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---

## 🛠 Features

- Text cleaning and preprocessing (lowercase, remove stopwords, punctuation, etc.)
- Tokenization and padding
- Deep learning model training & evaluation
- Real-time prediction interface via Streamlit
- GitHub version control and collaboration

---

