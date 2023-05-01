from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Embedding
from keras.utils import to_categorical, pad_sequences
from keras.layers import LSTM


# line by line sequence

# source text 
data = """
She had the jitters\n
She had the flu\n
She showed up late\n
She missed her cue\n
She kicked the director\n
She screamed at the crew\n
And tripped on a prop\n
And fell in some goo\n
And ripped her costume\n
A place or two\n
Then she forgot\n
A line she knew\n
And went “Meow”\n
Instead of “Moo”\n
She heard em giggle\n
She heard em boo\n
The programs sailed\n
The popcorn flew\n
As she stomped offstage\n
With a boo-hoo-hoo\n
The fringe of the curtain\n
Got caught in her shoe\n
The set crashed down\n
The lights did too\n
Maybe thats why she didnt want to do\n
An interview.\n """

#integer encode text
#tokenizer is dit on source text to develop mapping from words
#into unique integers, sequence of text can be converted to sequences
#of integers by calling texts_to__sequences() function
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

#determine vocabulary size [returns 73]
vocab_size = len(tokenizer.word_index) + 1

# sequence of integers, line-by-line
sequences = []
for line in data.split('\n'):
  encoded = tokenizer.texts_to_sequences([line])[0]
  for i in range(1, len(encoded)):
    sequence = encoded[:i+1]
    sequences.append(sequence)
print('total sequences: %d' % len(sequences))

# pad input sequences
# from keras, involves finding longest sequence, using that as length to pad other sequences
max_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_length, padding='pre')
print('max sequence length: %d' % max_length)

# split into input and output elements
sequences = np.array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)

# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=max_length-1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
model.summary()

# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit network
model.fit(X, y, epochs=500, verbose=2)

# generate a sequence from a language model
def generate_seq(model, tokenizer, max_length, seed_text, n_words):
  in_text = seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    # pre pad sequences to a fixed length
    encoded = pad_sequences([encoded], maxlen=max_length, padding='pre')
    # predict probabilities for each word
    probs = model.predict(encoded, verbose=0)
    yhat = np.argmax(probs)
    # map predicted word index to word
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    # append to input
    in_text += ' ' + out_word
  return in_text

# evaluate model
print("sequence with she:", generate_seq(model, tokenizer, max_length-1, 'She', 4))
print("sequence with the:",generate_seq(model, tokenizer, max_length-1, 'The', 4))
