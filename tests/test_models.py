import sys
import os


# TODO: Fix sibling directory imports
cwd = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(cwd, '..'))

from datautils.dataset import TextFile, Alphabet, _TOKEN_UNK, _TOKEN_PAD, _ID_UNK, _ID_PAD
from models.ngram import Ngram_LM

# TODO: Test predicting text probability should be independent of the number of timesteps per batch

# TODO: Test save/load model

# TODO: Test can reproduce training results


def test_ngram_consistent():
    filename = os.path.join(cwd, 'non-ascii-text.txt')
    with open(filename, 'r', encoding='utf-8') as fp:
        text = fp.read()

    text_train = text[:int(len(text) / 2)]
    text_test = text[int(len(text) / 2):]

    # Make sure there are unseen characters in the test text
    assert (not set(text_test).issubset(set(text_train)))

    train_filename = os.path.join(cwd, 'non-ascii-text-train.txt')
    test_filename = os.path.join(cwd, 'non-ascii-text-test.txt')

    with open(train_filename, 'w', encoding='utf-8') as fp:
        fp.write(text_train)

    with open(test_filename, 'w', encoding='utf-8') as fp:
        fp.write(text_test)

    train = TextFile(train_filename)

    alph = Alphabet.from_text(train_filename)

    for n in range(1, 8):
        model = Ngram_LM(n, alph)
        model.train(train)
        for i in range(1, len(test_filename)):
            probs = model.get_distribution(text_test, delta_smoothing=0.01)
            assert abs(1 - sum(probs.values())) < 1e6
