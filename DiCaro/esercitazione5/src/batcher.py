import os
import data
import numpy as np
import queue
import time
import warnings
from random import shuffle
from threading import Thread


class Example(object):
    """Represent a single input example instance of the NN 
    """

    def __init__(self, band_name, vocab, hps):

        band_name = band_name.split(' ')

        id_seq = [vocab.char2id(w) for w in band_name]

        if len(id_seq) > hps.seq_len:
            id_seq = id_seq[:hps.seq_len]

        input_seq = id_seq[:-1]
        target_seq = id_seq[1:]

        self.input_seq = input_seq
        self.input_length = len(input_seq)

        self.target_seq = target_seq
        self.target_length = len(target_seq)

    def pad_input(self, hps, pad_id):
        """Padding since NN input is fixed

        Args:
            hps ([type]): [description]
            pad_id ([type]): [description]
        """
        while len(self.input_seq) < hps.seq_len:
            self.input_seq.append(pad_id)

    def pad_output(self, hps, pad_id):

        while len(self.target_seq) < hps.seq_len:
            self.target_seq.append(pad_id)


class Batch(object):

    def __init__(self, example_list, vocab, hps):

        self.pad_id = vocab.char2id('\s')
        self.num_labels = vocab.size
        self.init_input(example_list, hps)
        self.init_target(example_list, hps)

    def init_input(self, example_list, hps):

        self.input = np.zeros([hps.batch_size, hps.seq_len], dtype=np.int32)
        self.input_mask = np.zeros([hps.batch_size, hps.seq_len], dtype=np.float32)
        self.input_length = np.zeros([hps.batch_size], dtype=np.int32)

        for i, ex in enumerate(example_list):

            ex.pad_input(hps, self.pad_id)

            self.input[i, :] = ex.input_seq[:]
            self.input_length[i] = ex.input_length

            for j in range(ex.input_length):
                self.input_mask[i,j] = 1

    def init_target(self, example_list, hps):

        self.target = np.zeros([hps.batch_size, hps.seq_len, hps.vocab_size], dtype=np.int32)
        self.target_mask = np.zeros([hps.batch_size, hps.seq_len], dtype=np.float32)
        self.target_length = np.zeros([hps.batch_size], dtype=np.int32)

        for i, ex in enumerate(example_list):

            ex.pad_output(hps, self.pad_id)

            self.target_length[i] = ex.target_length

            for j in range(ex.target_length):
                self.target[i, j, ex.target_seq[j]] = 1
                self.target_mask[i,j] = 1


class Batcher(object):

    def __init__(self, data_path, vocab, hps, max_queue_batch=100):

        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps

        # variable to generate the batch for train or test
        # in train case, the batches are generated infinitely

        self._max_queue_batch = max_queue_batch

        self._batch_queue = queue.Queue(self._max_queue_batch)
        self._example_queue = queue.Queue(self._max_queue_batch * self._hps.batch_size)

        self._num_example_q_threads = 16
        self._num_batch_q_threads = 4
        self._bucketing_chache_size = 100

        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):

            self._example_q_threads.append(Thread(target=self.fill_example_queue))
            self._example_q_threads[-1].daemon = True
            self._example_q_threads[-1].start()

        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            self._batch_q_threads.append(Thread(target=self.fill_batch_queue))
            self._batch_q_threads[-1].daemon = True
            self._batch_q_threads[-1].start()

        self._watch_thread = Thread(target=self.watch_threads)
        self._watch_thread.daemon = True
        self._watch_thread.start()

    def next_batch(self):

        if self._batch_queue.qsize() == 0:
            warnings.warn("Bucket input queue empty. Bucket queue size: %i, Input queue size: %i"
                          % (self._batch_queue.qsize(), self._example_queue.qsize()))

        batch = self._batch_queue.get()
        return batch

    def fill_batch_queue(self):

        while True:

            inputs = []

            for _ in range(self._hps.batch_size * self._bucketing_chache_size):
                inputs.append(self._example_queue.get())

            batches = []
            for i in range(0, len(inputs), self._hps.batch_size):
                batches.append(inputs[i:i+self._hps.batch_size])

            shuffle(batches)
            for b in batches:
                self._batch_queue.put(Batch(b, self._vocab, self._hps))

    def fill_example_queue(self):

        input_gen = self.text_generator(data.example_generator(self._data_path))

        while True:

            try:

                band_name = next(input_gen)

            except StopIteration:

                print("The example generator has exhausted saved_data")

                raise Exception("single_pass off: Error! example generator out of saved_data.")

            example = Example(band_name, self._vocab, self._hps)
            self._example_queue.put(example)

    def watch_threads(self):

        while True:

            time.sleep(60)

            for idx, t in enumerate(self._example_q_threads):

                if not t.is_alive():

                    print('\033[91m' + 'example thread dead. Restarting.' + '\033[0m')

                    new_t = Thread(target=self.fill_example_queue)
                    self._example_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

            for idx, t in enumerate(self._batch_q_threads):

                if not t.is_alive():

                    print('\033[91m' + 'batch thread dead. Restarting.' + '\033[0m')

                    new_t = Thread(target=self.fill_batch_queue)
                    self._batch_q_threads[idx] = new_t
                    new_t.daemon = True
                    new_t.start()

    def text_generator(self, example_generator):

        while True:

            e = next(example_generator)

            try:

                band_name = e.features.feature["char_string"].bytes_list.value[0].decode("utf-8")

            except ValueError:

                print('\033[91m' + 'Impossibile ritornare il nome della band' + '\033[0m')
                continue

            if len(band_name) == 0:

                warnings.warn("Trovato un esempio con nome band vuoto")
                continue

            else:

                yield band_name
