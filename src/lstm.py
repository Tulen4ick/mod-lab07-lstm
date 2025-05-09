import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import RMSprop
import random


def sample_index(preds, temperature = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length: int, diversity: float, words: list[str], max_length: int, vocabulary: list[str], words_to_indices: dict[str, int], indices_to_words: dict[int, str], model: Sequential):
    start_index = random.randint(0, len(words) - max_length - 1)
    sentence = words[start_index: start_index + max_length]
    generated = sentence.copy()
    for i in range(length):
        x_pred = np.zeros((1, max_length, len(vocabulary)))
        for t, char in enumerate(sentence):
            x_pred[0, t, words_to_indices[char]] = 1.
 
        preds = model.predict(x_pred, verbose = 0)[0]
        next_index = sample_index(preds, diversity)
        next_word = indices_to_words[next_index]
 
        generated.append(next_word)
        sentence = sentence[1:] + [next_word]
    return ' '.join(generated)


text = open('src/input.txt', 'r', encoding='utf-8').read().lower()
words = text.split()

vocabulary = sorted(list(set(words)))

words_to_indices = {word: i for i, word in enumerate(vocabulary)}
indices_to_words = {i: word for i, word in enumerate(vocabulary)}

max_length = 10
steps = 1
sentences = []
next_words = []

for i in range(0, len(words) - max_length, steps):
    sentences.append(words[i: i + max_length])
    next_words.append(words[i + max_length])

X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence):
        X[i, t, words_to_indices[word]] = 1
    y[i, words_to_indices[next_words[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape =(max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(learning_rate = 0.01)
model.compile(loss ='categorical_crossentropy', optimizer = optimizer)

model.fit(X, y, batch_size = 128, epochs = 50)

final_text = generate_text(1500, 0.2, words, max_length, vocabulary, words_to_indices, indices_to_words, model)
open('result/gen.txt', 'w', encoding='utf-8').write(final_text)
print(final_text)