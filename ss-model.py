from tensorflow import keras
import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers.core import Dense, Embedding
from keras.utils import to_categorical
from keras.layers import LSTM

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



# creating one-word-in, one-word-out sequence

#integer encode text
#tokenizer is dit on source text to develop mapping from words
#into unique integers, sequence of text can be converted to sequences
#of integers by calling texts_to__sequences() function
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])
encoded = tokenizer.texts_to_sequences([data])[0]

#determine vocabulary size [returns 73]
vocab_size = len(tokenizer.word_index) + 1

# create sequences of words to fit the model
# with one word as input and one word as output

#create word -> word sequences [106]
sequences = []

for i in range(1, len(encoded)):
  sequence = encoded[i-1:i+1]
  sequences.append(sequence)

# split elements into input (x) and output (y)
# use numpy array feature to split the data into input and output elements using indexing tuples
# first element of each sequence as input, second element as output
sequences = np.array(sequences)
x, y = sequences[:,0],sequences[:,1]
# print("Sequences:", sequences)
# model is fitted to predict probability distribution of all words in
# vocabulary, need to turn output element from single integer
# into one hot encoding with a 0 for every word in the vocabulay
# and one for every actual word of that value 
# creates "ground truth" for network to aim for to calculate error & update model

#keras to_categorical() function can convert integer to a one hot encoding
# while specifying number of calsses as the vocabulary size

y = to_categorical(y, num_classes=vocab_size)

#model uses a learned word embedding in the input layer
#one real-valued vector for each word in the vocabulary
#each word vector is of a specified length
# 10-dimensional projection
# input sequence contains a single word, input_length = 1

#define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=1))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))

# print each layer of model
# for i, layer in enumerate(model.layers):
#   print(f'Layer {i}: {layer.name}, Type: {type(layer)}, Trainable: {layer.trainable}, output shape: {layer.output_shape}')

# fitting network on encoded text data
# technically we are modeling multi-class classifcation problem
# (predict work in vocaubulary) using categorial cross entropy loss function
# we use the efficient adam implementation of gradient descent and track accuracy at the end of each epoch
# the model is fit for 500 training epochs, though more may be needed

# compile and fit network to predict word in vocabulary
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
model.fit(x, y, epochs=500, verbose=2)
print('Vocabulary Size: %d' % vocab_size)
print('Total Sequences: %d' % len(sequences))
print("x", x, "y", y)
print("MODEL SUMMARY")
model.summary()

# test by passing in a given word from vocabulary and predicting next word
# in_text = 'She'
# encoded = tokenizer.texts_to_sequences([in_text])[0]
# encoded = np.array(encoded)
# probs = model.predict(encoded, verbose=0)
# yhat = np.argmax(probs)
# for word, index in tokenizer.word_index.items():
#   if index == yhat:
#     print('in text word:', in_text)
#     print('predicted word:', word)

# generate a sequence from the model
def generate_sequence(model, tokenizer, seed_text, n_words):
  in_text, result = seed_text, seed_text
  # generate a fixed number of words
  for _ in range(n_words):
    # encode the text as integer
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = np.array(encoded)
    # predict a word in the vocabulary
    probs = model.predict(encoded, verbose=0)
    yhat = np.argmax(probs)
    # map predicted word index to word
    out_word = ''
    for word, index in tokenizer.word_index.items():
      if index == yhat:
        out_word = word
        break
    # append to input
    in_text, result = out_word, result + ' ' + out_word
    return result

# evaluate model
print("generated sequence:", generate_sequence(model, tokenizer, 'She', 6))