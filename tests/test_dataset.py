import sys
import os
import numpy as np

# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd, '..'))

from datautils.dataset import TextFile, Alphabet, _TOKEN_UNK, _TOKEN_PAD, _ID_UNK, _ID_PAD, Dataset


def pytest_generate_tests(metafunc):
    # called once per each test function
    funcarglist = metafunc.cls.params[metafunc.function.__name__]
    argnames = sorted(funcarglist[0])
    metafunc.parametrize(argnames, [[funcargs[name] for name in argnames]
            for funcargs in funcarglist])


configs = [
        dict(memory_limit=int(5e5), batch_size=1, num_timesteps=5),
        dict(memory_limit=int(10e5), batch_size=1, num_timesteps=100),
        dict(memory_limit=int(10e11), batch_size=2, num_timesteps=10),
        dict(memory_limit=int(10e9), batch_size=3, num_timesteps=1000)
    ]

class TestDataset:
    params = {
        'test_inputs_and_targets_match': configs,
        'test_can_reconstruct_text': configs,
        'test_inputs_and_targets_have_correct_shapes': configs
    }

    def setup_method(self):
        self.filename = os.path.join(cwd, 'non-ascii-text.txt')
        self.alphabet = Alphabet.from_text(self.filename)
        with open(self.filename, 'r') as fp:
            self.text = fp.read()

    def teardown_method(self):
        self.filename = None
        self.alphabet = None
        self.text = None

    def test_inputs_and_targets_have_correct_shapes(self, memory_limit, batch_size, num_timesteps):
        ds = Dataset(self.filename, self.alphabet, memory_limit)
        it = ds.get_batch_iterator(batch_size, num_timesteps)

        num_batches = 0
        for inputs, targets in it:
            num_batches += 1
            assert inputs.shape[0] == batch_size
            assert inputs.shape[1] <= num_timesteps
            assert inputs.shape[2] == self.alphabet.get_size()
            assert targets.shape[0] == batch_size
            assert targets.shape[1] <= num_timesteps

        if len(self.text) > 0:
            assert num_batches > 0

    def test_inputs_and_targets_match(self, memory_limit, batch_size, num_timesteps):
        ds = Dataset(self.filename, self.alphabet, memory_limit)
        it = ds.get_batch_iterator(batch_size, num_timesteps)

        num_batches = 0
        for inputs, targets in it:
            num_batches += 1
            inp_ids = inputs.argmax(axis=2)
            for inp_sequence, tar_sequence in zip(inp_ids, targets):
                assert np.all(inp_sequence[1:] == tar_sequence[:-1])

        if len(self.text) > 0:
            assert num_batches > 0
    #
    def test_can_reconstruct_text(self, memory_limit, batch_size, num_timesteps):
        """
        Will fail if the memory limit is small enough so that the text has to be read in chunks
        """
        ds = Dataset(self.filename, self.alphabet, memory_limit)

        it = ds.get_batch_iterator(batch_size, num_timesteps)

        text_chunks = [''] * batch_size

        for inputs, targets in it:
            for i, sequence in enumerate(targets):
                text_chunks[i] += self.alphabet.ids_to_string(sequence)

        reconstructed_text = ''.join(text_chunks)

        # Our data provider will truncate the input so that it is a multiple of the batch_size
        assert reconstructed_text == self.text[:len(self.text) - len(self.text) % batch_size]

