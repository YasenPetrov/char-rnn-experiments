import sys
import os

# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd, '..'))

from datautils.dataset import TextFile, Alphabet, _TOKEN_UNK, _TOKEN_PAD, _ID_UNK, _ID_PAD


def test_text_iterator_reads_non_ascii_correctly():
    """
    Read a file normally and by iterating over it, make sure we end up with the same result
    """
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    with open(filename, 'r', encoding='utf-8') as fp:
        text = fp.read()

    file = TextFile(filename)

    # Read in blocks of 100 chars
    it = file.get_iterator(100)

    text_read = file.read_whole_file()

    text_iterated = ''
    for chunk in it:
        text_iterated += chunk

    assert text_read == text
    assert text_iterated == text


def test_text_iterator_chunks_regular_size():
    """
    Make sure our iterator returns chunks of the same size
    """
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    file = TextFile(filename)

    chunk_size = 11
    # Read in blocks of chunk_size chars
    it = file.get_iterator(chunk_size)

    # We can only allow for the last chunk in the file to have a length < chunk_size
    already_read_chunk = False
    last_chunk_size = None
    for chunk in it:
        if already_read_chunk:
            assert last_chunk_size == chunk_size
        already_read_chunk = True
        last_chunk_size = len(chunk)


def test_test_iterator_reset():
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    file = TextFile(filename)

    texts = []

    for _ in range(2):
        chunk_size = 1000
        # Read in blocks of chunk_size chars
        it = file.get_iterator(chunk_size)

        text_iterated = ''
        for chunk in it:
            text_iterated += chunk
        texts.append(text_iterated)

    assert texts[0] == texts[1]


def is_valid_alphabet(alph):
    """
    :param alph: an Alphabet object
    """

    # Make sure we have special characters
    assert alph.int_to_char[_ID_PAD] == _TOKEN_PAD
    assert alph.int_to_char[_ID_UNK] == _TOKEN_UNK

    # Make sure we have unique IDs
    assert len(set(alph.int_to_char.keys())) == len(alph.int_to_char.keys())

    # Make sure we have the same sets of ids and characters
    assert set(alph.char_to_int.keys()) == set(alph.int_to_char.values())
    assert set(alph.int_to_char.keys()) == set(alph.char_to_int.values())

    return True


def test_alphabet_creation():
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    alph = Alphabet.from_text(filename)

    assert is_valid_alphabet(alph)

    # Make sure we have the same character set as the original text
    alph_set = set(alph.char_to_int.keys())

    with open(filename, 'r', encoding='utf-8') as fp:
        real_set = set(fp.read())

    # Remove special characters before comparison
    alph_set.remove(_TOKEN_PAD)
    alph_set.remove(_TOKEN_UNK)
    assert alph_set == real_set


def test_alphabet_save_load():
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    alph = Alphabet.from_text(filename)

    assert is_valid_alphabet(alph)

    alphabet_filename = os.path.join(cwd, 'test_alphabet.json')
    alph.dump_to_json(alphabet_filename)

    assert os.path.exists(alphabet_filename)

    alph_loaded = Alphabet.from_json(alphabet_filename)

    assert is_valid_alphabet(alph_loaded)
    assert alph.char_to_int == alph_loaded.char_to_int


def test_alphabet_tokenization():
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    alph = Alphabet.from_text(filename)

    assert is_valid_alphabet(alph)

    with open(filename, 'r') as fp:
        text = fp.read()

    ids = alph.string_to_ids(text)

    assert len(ids) == len(text)

    converted_text = alph.ids_to_string(ids)

    assert text == converted_text

    converted_ids = alph.string_to_ids(converted_text)

    for i, id in enumerate(ids):
        assert id == converted_ids[i]


