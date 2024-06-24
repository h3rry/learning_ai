from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Preprocessing
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_texts)
sequences = tokenizer.texts_to_sequences(train_texts)
data = pad_sequences(sequences, maxlen=500)

# Model building
model = Sequential()
model.add(Embedding(5000, 32, input_length=500))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))

# Model compilation
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

# Model training
model.fit(data, train_labels, epochs=10, batch_size=128, validation_split=0.2)

# Sentiment prediction
test_sequences = tokenizer.texts_to_sequences(["Tidak ada yang membuat saya senang, tertawa, dan bahagia dalam film ini."])
test_data = pad_sequences(test_sequences, maxlen=500)
prediction = model.predict(test_data)