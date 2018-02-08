import os
import json
import numpy as np
# from lmexperiments.config import PROJECT_HOME

# Special IDs for unknown characters and padding
_ID_UNK = 0
_ID_PAD = 1
_TOKEN_UNK = u'\ufffd'
_TOKEN_PAD = u'\u0080'


# Roughly how much memory we want an array of one-hot encoded to occupy, at most
_MEMORY_LIMIT_BYTES = int(1e9)


class TextFile:
    """
    Encompasses actions required to read from a potentially very large text file
    """
    def __init__(self, filename, encoding='utf-8'):
        # Make sure file exists
        if not os.path.exists(filename):
            raise FileNotFoundError(filename)

        self.filename = filename
        self.encoding = encoding

    def get_size_in_bytes(self):
        return os.path.getsize(self.filename)

    def read_whole_file(self):
        with open(self.filename, 'r') as fp:
            text = fp.read()
        return text

    def get_iterator(self, max_chunk_size=int(1e6)):
        """
            Returns a generator object that iterates over a large file in chunks
            :param filename: Path to the text file
            :param max_chunk_size: Maximum number of chars to be read in at a time
            :return: A generator object
            """
        with open(self.filename, 'r', encoding=self.encoding) as fp:
            chunk = fp.read(max_chunk_size)
            while chunk:
                yield chunk
                chunk = fp.read(max_chunk_size)


class Alphabet:
    def __init__(self, char_to_int):
        self.char_to_int = char_to_int
        self.int_to_char = dict((i, c) for c, i in char_to_int.items())

    def string_to_ids(self, s):
        ids = []
        for c in s:
            if c in self.char_to_int:
                ids.append(self.char_to_int[c])
            else:
                ids.append(_ID_UNK)
        return ids

    def ids_to_string(self, ids):
        # TODO: Error handling
        s = ''.join([self.int_to_char[id] for id in ids])
        return s

    def dump_to_json(self, filename):
        if os.path.exists(filename):
            #TODO: Put warning here
            try:
                os.remove(filename)
            except:
                print('Could not remove file {0}'.format(filename))
        with open(filename, 'w+') as fp:
            json.dump(self.char_to_int, fp, indent=2)

    def get_size(self):
        return len(self.char_to_int)

    def get_chars(self):
        return list(self.char_to_int.keys())

    def filter_string(self, text):
        return self.ids_to_string(self.string_to_ids(text))

    @staticmethod
    def from_text(filename):
        file = TextFile(filename)

        chars = set()

        # Read 10 mln chars at a time
        for chunk in file.get_iterator(int(1e7)):
            chars.update(chunk)

        # make sure we remove reserved characters from the set
        if _TOKEN_PAD in chars: chars.remove(_TOKEN_PAD)
        if _TOKEN_UNK in chars: chars.remove(_TOKEN_UNK)

        char_to_int = {
            _TOKEN_PAD : _ID_PAD,
            _TOKEN_UNK : _ID_UNK
        }

        # Fill the rest of the map with the characters from the text, making sure they do not overwrtie special chars
        next_id = len(char_to_int)
        for c in sorted(chars):
            char_to_int[c] = next_id
            next_id += 1

        return Alphabet(char_to_int)

    @staticmethod
    def from_json(filename):
        with open(filename, 'r', encoding='utf-8') as fp:
            char_to_int = json.load(fp)

        if _TOKEN_UNK not in char_to_int or _TOKEN_PAD not in char_to_int or not char_to_int[_TOKEN_UNK] ==_ID_UNK or \
                not char_to_int[_TOKEN_PAD] == _ID_PAD:
            raise ValueError('''While loading Alphabet from {2}: Alphabet should contain keys {0} and {1}, correspondin\
             g to ids {3} and {4}'''.format(_TOKEN_PAD, _TOKEN_UNK, filename, _ID_PAD, _ID_UNK))

        return Alphabet(char_to_int)


def to_categorical(ids, alphabet_size, add_padding=True, padding_value=_ID_PAD):
    """
    Convert a list of integer ids to a one-hot representation
    :param ids(list[int]): A list of integer ids
    :param alphabet_size(int): The size of the encoding alphabet - this determines the length
             of the one-hot encoded vectors
    :return: np.array of size (len(ids), alphabet_size) with each row corresponding to a one-hot encoded character
    """
    result = np.zeros((len(ids), alphabet_size))

    if add_padding: # Add a padding token to the start of the sequence
        result[0][padding_value] = 1
        for i, id in enumerate(ids[:-1]):
            result[i + 1][id] = 1
    else:
        for i, id in enumerate(ids):
            result[i][id] = 1
    return result


class Dataset:
    def __init__(self, path, alphabet, memory_limit_bytes=_MEMORY_LIMIT_BYTES):
        self.alphabet = alphabet
        self.path = path
        self._textfile = TextFile(path)
        self._inputs_data = None
        self._targets_data = None
        self.memory_limit_bytes = memory_limit_bytes

    def get_batch_iterator(self, batch_size, num_timesteps):
        #TODO: Support for output in the form (num_timesteps x num_batches x alphabet_size)
        # Decide on an appropriate chunk size for reading text we will be reading
        max_chars = self.memory_limit_bytes // (8 * self.alphabet.get_size())
        # Make chunk size a multiple of (timesteps x batch_size)
        max_chunk_size = max_chars - (max_chars % (batch_size * num_timesteps))

        if max_chunk_size < 1:
            raise MemoryError('Cannot fit a batch of size {0} of sequences of size {1} in {2} bytes of memory'.format(
                batch_size, num_timesteps, self.memory_limit_bytes
            ))

        # Keep track of the last id in the previous chunk
        last_chunk_last_id = _ID_PAD

        # Iterate over chunks of the file
        for chunk in self._textfile.get_iterator(max_chunk_size):
            # Turn chunk into list of ids
            ids = self.alphabet.string_to_ids(chunk)

            # If this data provider will be used with pyTorch RNNs - these models take input of form
            # <batch_size x num_timesteps x alphabet_size>, where num_timesteps can vary but batch_size cannot
            # When we get the last chunk, chances are its length won't be a multiple of (batch_size * num_timesteps),
            # so we won't be able to divide it into properly sized batches. We have made max_chunk_size to be a multiple
            # of (batch_size * num_timesteps) so we needn't worry about any chunks but the last one
            # So for our last chunk, we truncate it so that its length is a multiple of the batch_size - the
            # Truncated version is of length (M * batch_size), so we just return inputs and targets of shapes
            # <batch_size x M x alphabet_size> and <batch_size, M>, respectively

            # How many timesteps we can fit, how many ids we have to throw out
            quotient, remainder = divmod(len(ids), batch_size)
            ids = ids[:len(ids) - remainder]

            if len(ids) > 0:
                # How many batches with the regular number of timesteps, how many timesteps in the last,odd-length batch
                num_reg_sized_batches, timesteps_in_last_batch = divmod(quotient, num_timesteps)

                # Make inputs, pad with last character of previous chunk and cut last character of this chunk
                # and reshape it into <batch_size, T, alphabet_size>, where T is the total number of timesteps that
                # fit in this chunk
                inputs_data = to_categorical(ids, self.alphabet.get_size(), True, last_chunk_last_id).reshape(
                    batch_size, (num_reg_sized_batches * num_timesteps + timesteps_in_last_batch), self.alphabet.get_size()
                )
                # Make targets (not one-hot encoded), reshape into <batch_size, T>
                targets_data = np.array(ids).reshape(
                    batch_size, (num_reg_sized_batches * num_timesteps + timesteps_in_last_batch)
                )

                # We will return slices from the data arrays - keep track of the start and end indices
                batch_start, batch_end = 0, num_timesteps

                # Loop over whole chunk, yield batches
                while batch_start < inputs_data.shape[1]:
                    yield inputs_data[:, batch_start:batch_end, :], targets_data[:, batch_start:batch_end]
                    batch_start += num_timesteps
                    # As described above, the last batch can have a sequence of a different length
                    batch_end = min(inputs_data.shape[1], batch_end + num_timesteps)


