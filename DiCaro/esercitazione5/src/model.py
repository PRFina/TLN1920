from collections import namedtuple
from data import Vocabulary
from batcher import Batcher
from keras.models import Sequential
from keras.layers import GRU, Dense, Embedding, Dropout
from keras.callbacks import LambdaCallback
from beam_search import beam_search

import numpy as np


vocab = Vocabulary("./vocabulary.txt")

# Hyperprameters
params = {
       'hid_dim': 128,
       'emb_dim': 64,
       'lr': 0.05,
       'vocab_size': vocab.size,
       'keep_prob': 0.8,
       'batch_size': 8,
       'beam_size': 5,
       'seq_len': 50,
       'optimizer': 'adagrad'}

param_list = list(params.keys())

hps = namedtuple("PARAM", param_list)(**params)


batcher = Batcher('./train.bin', vocab, hps)

# Model Architecture
model = Sequential()
model.add(Embedding(vocab.size, hps.emb_dim, input_length=hps.seq_len))
model.add(Dropout(1.-hps.keep_prob))
model.add(GRU(hps.hid_dim, return_sequences=True, unroll=True))
model.add(Dense(vocab.size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=hps.optimizer)


def make_name(model, vocab, hps):
       """ generate output name using greedy search

       Args:
           model (): model architecture specification
           vocab (): vocabulary
           hps (): hyperparameters
       """
       name = []
       x = np.ones((1, hps.seq_len)) * vocab.char2id('<s>')
       i = 0

       # generate chars until seq. max lenght or <\s> end token selection
       while i < hps.seq_len:

              probs = list(model.predict(x)[0, i])
              probs = probs / np.sum(probs)
              index = np.random.choice(range(vocab.size), p=probs)
              character = vocab.id2char(index)

              if character == '\s':
                     name.append(' ')
              else:
                     name.append(character)

              if i >= hps.seq_len or character == '</s>':
                     break
              else:
                     x[0, i + 1] = index

              i += 1

       print(''.join(name))


def make_name_beam(model, vocab, hps):

       best_seq = beam_search(model, vocab, hps)
       chars = [vocab.id2char(t) for t in best_seq.tokens[1:]]
       tokens = [t if t != '\s' else ' ' for t in chars]
       tokens = ''.join(tokens)

       print(tokens)

# Training loop
iteration = 0
while True:

       batch = batcher.next_batch()

       model.train_on_batch(batch.input, batch.target)

       if iteration % 1000 == 0:

              print('Names generated after iteration %d:' % iteration)

              for i in range(3):
                     make_name(model, vocab, hps)

              print()

       iteration += 1
