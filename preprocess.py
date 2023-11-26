import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import pandas as pd

data = pd.read_csv('medium_data.csv')
data = data['title']

# Removing unwanted characters and words
data = data.apply(lambda x: x.replace(u'\xa0',u' '))
data = data.apply(lambda x: x.replace('\u200a',' '))

# Tokenizing
tokenizer = Tokenizer(oov_token='<oov>') # For those words which are not found in word_index
tokenizer.fit_on_texts(data)
total_words = len(tokenizer.word_index) + 1

print("Total number of words: ", total_words)
print("Word: ID")
print("------------")
print("<oov>: ", tokenizer.word_index['<oov>'])
print("Strong: ", tokenizer.word_index['strong'])
print("And: ", tokenizer.word_index['and'])
print("Consumption: ", tokenizer.word_index['consumption'])

input_sentences = []

for sentence in data:
    print(sentence)
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

    for i in range(1, len(tokenized_sentence)):
        input_sentences.append(tokenized_sentence[:i+1])

max_length = max([len(x) for x in input_sentences])
total_word =  len(tokenizer.word_index) + 1 
# print(max_length)

padded_input_sequences = pad_sequences(input_sentences, maxlen = max_length, padding = 'pre')

X = padded_input_sequences[:,:-1] 
y = padded_input_sequences[:,-1]
y = to_categorical(y, num_classes=total_word)

