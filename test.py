from preprocess import *
from model import *
from train import * 
import numpy as np

text =  "human"
next_words =  2
for _ in range(next_words):
    token_text = tokenizer.texts_to_sequences([text])[0]
    padded_token_text = pad_sequences([token_text], maxlen = max_length-1, padding = 'pre')
    prediction = np.argmax(model.predict(padded_token_text))
    # prediction = model.predict_classes(padded_token_text)
    text += " " + tokenizer.index_word[prediction]

