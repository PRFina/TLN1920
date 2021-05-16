import glob
import random
import struct
from tensorflow.core.example import example_pb2


def example_generator(data_path):
    """Unpack tfrecords file and yield one train example at time

    Args:
        data_path (str): path to the tfrecords file

    Yields:
        [type]: [description]
    """
    while True:

        filelist = glob.glob(data_path)
        assert filelist, ("Error: Empty filelist at %s" % data_path)

        random.shuffle(filelist)

        for f in filelist:
            reader = open(f, "rb")
            while True:
                len_bytes = reader.read(8)
                if not len_bytes: break
                str_len = struct.unpack("q", len_bytes)[0]
                example_str = struct.unpack("%ds" % str_len, reader.read(str_len))[0]
                yield example_pb2.Example.FromString(example_str)


class Vocabulary(object):
    """Class to represent a mapping between vocabulary lemma (a char) and'its integer index
    """

    def __init__(self, voc_file, voc_size=400):

        self._char2id = {}
        self._id2char = {}
        self._size = 0

        with open(voc_file, 'r', encoding='utf-8', errors='ignore') as reader:

            i = 0
            for row in reader:

                char_freq = row.strip().split()

                if len(char_freq) > 2:
                    print("Non e' una singola parola")
                    continue

                char = char_freq[0]

                if char in self._char2id:
                    print(f"carattere {char} gia' presente")
                    continue
                # create mappping char-index
                self._char2id[char] = i
                self._id2char[i] = char
                i += 1

                if i == voc_size:
                    print(f"raggiunta dimensione massima vocabolario. Ultima parola inserita: {self._id2char[i-1]}")
                    break

            self._size = i


    def char2id(self, char):
        """get the index of the char

        Args:
            char (str): character

        Returns:
            [int]: index of the character
        """
        return self._char2id[char]

    def id2char(self, id):
        """get the index corresponding character

        Args:
            id (int): index

        Returns:
            [str]: character
        """

        if id not in self._id2char:
            raise ValueError(f'id {id} non presente nel vocabolario')

        return self._id2char[id]

    @property
    def size(self):
        return self._size


if __name__ == '__main__':

    vocab = Vocabulary('vocabulary.txt', 400)
    print(vocab.size)
