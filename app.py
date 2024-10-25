
import numpy as np
import tensorflow
from tensorflow import keras 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding,Dense,SimpleRNN
from keras.models import load_model
import streamlit as st

model = load_model('SimpleRNN.h5')

st.title("Fake News Detection")

sentence = st.text_area("Enter a sentence")
def predict_sentiment(sentence):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([sentence])
    sequences = tokenizer.texts_to_sequences([sentence])
    padded_sequences = pad_sequences(sequences, maxlen=500)
    prediction = model.predict(padded_sequences)
    return prediction


if st.button("Predict"):
    if len(sentence) != 0:
       prediction = predict_sentiment(sentence)
    
       if prediction[0] > 0.5:
            st.success("The News is Real")
       else:
            st.error("The News is Fake")

