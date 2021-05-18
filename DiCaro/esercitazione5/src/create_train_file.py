import csv
import struct
from collections import Counter
from tensorflow.core.example import example_pb2


def sent2char(sent):
    s = ['<b>'] + [c if c != ' ' else '\s' for c in sent] + ['</b>']
    return s


def simple_sent2char(sent):
    return [str(c) if c != ' ' else '\s' for c in sent]


def read_csv_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as fin:
        csv_reader = csv.DictReader(fin)

        for row in csv_reader:
            yield row


def extract_data(filename):
    data = []
    for row in read_csv_file(filename):
        # char_string = ['<s>', '\s'] + sent2char(row['name']) \
        #              + sent2char(row['genre']) + sent2char(row['country']) + ['</s>']
        char_string = ['<s>'] + simple_sent2char(row['name']) + ['</s>']

        data.append(char_string)

    return data


def generate_vocabulary(data):
    counter = Counter()
    for line in data:
        counter.update(line)

    with open('vocabulary.txt', 'w') as writer:
        for (c, f) in counter.most_common(1000):
            writer.write(f"{c}\t{str(f)}\n")


def create_train_file(data):
    with open('train.bin', 'wb') as writer:
        for char_string in data:
            tf_example = example_pb2.Example()
            encoded_string = ' '.join(char_string).encode('utf-8')
            tf_example.features.feature["char_string"].bytes_list.value.extend([encoded_string])
            tf_example_str = tf_example.SerializeToString()
            str_len = len(tf_example_str)
            writer.write(struct.pack('q', str_len))
            writer.write(struct.pack('%ds' % str_len, tf_example_str))


if __name__ == '__main__':
    # Se si vuole cambiare il dataset, è sufficente passare il Path qui.
    data = extract_data('starwars.csv')
    generate_vocabulary(data)
    create_train_file(data)
