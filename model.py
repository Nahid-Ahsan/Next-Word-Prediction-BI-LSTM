from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from preprocess import * 

model = Sequential()
model.add(Embedding(total_word,100, input_length = max_length-1))
model.add(Bidirectional(LSTM(150)))
model.add(Dense(total_word, activation = 'softmax',))

model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics = 'accuracy')

model.summary()



