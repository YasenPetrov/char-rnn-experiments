import os
import json
import re
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

    def read_whole_file(self, max_chars=None):
        with open(self.filename, 'r', encoding=self.encoding) as fp:
            if max_chars is None or max_chars < 0 or max_chars == np.inf:
                text = fp.read()
            else:
                text = fp.read(max_chars)
        return text

    def get_iterator(self, max_chunk_size=int(1e6), max_chars=np.inf):
        """
            Returns a generator object that iterates over a large file in chunks
            :param filename: Path to the text file
            :param max_chunk_size: Maximum number of chars to be read in at a time
            :param max_chars: Maximum number of chars to be returned in total

            :return: A generator object
            """
        chars_left = max_chars
        with open(self.filename, 'r', encoding=self.encoding) as fp:
            chunk = fp.read(min(max_chunk_size, chars_left))
            while chunk and chars_left > 0:
                chunk = chunk[:min(len(chunk), chars_left)]
                chars_left -= len(chunk)
                yield chunk
                chunk = fp.read(min(max_chunk_size, chars_left))


class BunchOfFiles:
    def __init__(self, filenames, alphabet, max_lengths=None, sampling_weights=None):
        for f in filenames:
            if not os.path.exists(f):
                raise FileNotFoundError(f'{f}')

        self.filenames = filenames
        self.alphabet = alphabet
        self.max_lengths = max_lengths
        self.sampling_weights = sampling_weights

        if self.max_lengths is None:
            self.max_lengths = [np.inf for _ in filenames]
        if self.sampling_weights is None:
            self.sampling_weights = [1 for _ in self.filenames]

        self.texts = self._read_files()

    def _read_files(self):
        texts = []
        for f, max_chars in zip(self.filenames, self.max_lengths):
            textfile = TextFile(f)
            texts.append(textfile.read_whole_file(max_chars))
        return texts

    def get_batch_iterator(self, batch_size, seq_length, separator=r'[\.\?!]\s+', remove_overlapping=True,
                           shuffle_sequences=False, max_chars=np.inf, pad_input=True):

        # Matp from text index to array of starts and a counter for how many sequences have been used
        # We need the latter so that we do not repeat sequences in our evaluation set
        sequence_map = [dict() for _ in range(len(self.texts))]
        for i, text in enumerate(self.texts):
            #Get the indices where sentences start
            sent_starts = [m.end() for m in re.finditer(separator, text) if m.end() + seq_length <= len(text)]

            if remove_overlapping and len(sent_starts):
                # We do not want overlapping sequences -- remove sentence starts that fall within the previous sequence
                s = [sent_starts[0]]
                last_start = s[0]
                for start in sent_starts[1:]:
                    if start > last_start + seq_length:
                        s.append(start)
                        last_start = start
                sent_starts = s

            if shuffle_sequences:
                np.random.shuffle(sent_starts)

            sequence_map[i]['sent_starts'] = sent_starts
            sequence_map[i]['sentences_used'] = 0

        # Fill with tuples of (text_index, sentence_start)
        text_start_tuples = []

        text_pool_indices = list(range(len(self.texts)))
        sampling_weights = np.array([w for w in self.sampling_weights], dtype='float')
        sampling_weights /= sampling_weights.sum()

        while len(text_start_tuples) * seq_length < max_chars and len(text_pool_indices) > 0:
            # Sample text index
            text_ix = np.random.choice(text_pool_indices, p=sampling_weights)
            # If text out of sequences, remove that text's index from the pool of texts and move on
            if sequence_map[text_ix]['sentences_used'] >= len(sequence_map[text_ix]['sent_starts']):
                ix_to_remove = text_pool_indices.index(text_ix)
                text_pool_indices.remove(text_ix)
                sampling_weights = np.delete(sampling_weights, ix_to_remove)
                sampling_weights /= sampling_weights.sum()
                continue
            # Get the next sequence from that text
            next_seq_start = sequence_map[text_ix]['sent_starts'][sequence_map[text_ix]['sentences_used']]
            sequence_map[text_ix]['sentences_used'] += 1
            text_start_tuples.append((text_ix, next_seq_start))

        # Make sure the number of seqs is a multiple of the batch_size
        seqs =[self.texts[ti][start: start + seq_length] for ti, start in text_pool_indices]
        remainder = len(seqs) % batch_size
        seqs = seqs[:len(seqs) - remainder]

        seqs = [self.alphabet.string_to_ids(s, remove_unknown=False) for s in seqs]

        if pad_input:
            inputs = np.array(
                [to_categorical(s, self.alphabet.get_size(), add_padding=pad_input) for s in seqs])
            targets = np.array(seqs)
        else:
            inputs = np.array(
                [to_categorical(s[:-1], self.alphabet.get_size(), add_padding=pad_input) for s in seqs])
            targets = np.array([s[1:] for s in seqs])

        batch_start = 0
        # Loop over whole chunk, yield batches
        while batch_start + batch_size <= inputs.shape[0]:
            yield inputs[batch_start: batch_start + batch_size], targets[batch_start:batch_start + batch_size]
            batch_start += batch_size


class Alphabet:
    def __init__(self, char_to_int):
        self.char_to_int = char_to_int
        self.int_to_char = dict((i, c) for c, i in char_to_int.items())

    def string_to_ids(self, s, remove_unknown=False):
        ids = []
        for c in s:
            if c in self.char_to_int:
                ids.append(self.char_to_int[c])
            else:
                if not remove_unknown:
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

    def get_batch_iterator(self, batch_size, num_timesteps, remove_unknown_tokens=False, num_chars_to_read=np.inf):
        # TODO: Support for output in the form (num_timesteps x num_batches x alphabet_size)
        # Decide on an appropriate chunk size for reading text we will be reading
        max_chars = self.memory_limit_bytes // (8 * self.alphabet.get_size())
        # Make chunk size a multiple of (timesteps x batch_size)
        max_chunk_size = max_chars - (max_chars % (batch_size * num_timesteps))
        # (Optional) Make sure the chunk size is a multiple of the number of timesteps we want to reset the hidden
        # state of our model after times the batch size-- when we load a new chunk, the sequence of characers
        # defined by the first character in every batch will not be a continuation of the same sequence from the
        # previous chunk. If we do not reset the hidden state, we will surprise our model
        max_chunk_size = max_chunk_size - max_chunk_size % (batch_size * num_timesteps)

        assert max_chunk_size > 0

        if max_chunk_size < 1:
            raise MemoryError('Cannot fit a batch of size {0} of sequences of size {1} in {2} bytes of memory'.format(
                batch_size, num_timesteps, self.memory_limit_bytes
            ))

        # Keep track of the last id in the previous chunk
        last_chunk_last_id = _ID_PAD

        # Iterate over chunks of the file
        for chunk in self._textfile.get_iterator(max_chunk_size, max_chars=num_chars_to_read):
            # Turn chunk into list of ids
            ids = self.alphabet.string_to_ids(chunk, remove_unknown=remove_unknown_tokens)

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

            last_chunk_last_id = ids[-1]


class SentenceDataset:
    def __init__(self, files_list, alphabet, memory_limit_bytes=_MEMORY_LIMIT_BYTES):
        self.alphabet = alphabet
        self.files_list = files_list
        self.memory_limit_bytes = memory_limit_bytes

    def get_batch_iterator(self, batch_size, num_timesteps, separator=r'[\.\?!]\s+', remove_unknown_tokens=False,
                           pad_input=False, max_chars_per_file=np.inf):

        for file in self.files_list:
            textfile = TextFile(file)
            for chunk in textfile.get_iterator(max_chars=max_chars_per_file):
                # Get the string sequences we are interested in (get one more character if we are not padding)
                seqs = [chunk[m.end(0): m.end(0) + num_timesteps + int(not pad_input)] for m in
                        re.finditer(separator, chunk)]

                # Chances are the last sequence will be of odd length -- remove it if it is
                while not len(seqs[-1]) == num_timesteps + int(not pad_input):
                    seqs = seqs[:-1]

                # Make sure the number of seqs is a multiple of the batch_size
                remainder = len(seqs) % batch_size
                seqs = seqs[:len(seqs) - remainder]

                seqs = [self.alphabet.string_to_ids(s, remove_unknown=False) for s in seqs]

                if pad_input:
                    inputs = np.array(
                        [to_categorical(s, self.alphabet.get_size(), add_padding=pad_input) for s in seqs])
                    targets = np.array(seqs)
                else:
                    inputs = np.array(
                        [to_categorical(s[:-1], self.alphabet.get_size(), add_padding=pad_input) for s in seqs])
                    targets = np.array([s[1:] for s in seqs])

                batch_start = 0
                # Loop over whole chunk, yield batches
                while batch_start + batch_size <= inputs.shape[0]:
                    yield inputs[batch_start: batch_start + batch_size], targets[batch_start:batch_start + batch_size]
                    batch_start += batch_size