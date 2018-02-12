import os
from itertools import product

from datautils.dataset import Alphabet

#TODO: Documentation

def dict_of_lists_to_list_of_dicts(dict_of_lists):
    combinations = product(*(dict_of_lists.values()))
    return [dict(zip(dict_of_lists.keys(), combo)) for combo in combinations]


def clean_and_split_file(source_file, destination_dir, train_prop, valid_prop, test_prop, use_first_n_characters=0,
                         alphabet_file=None):
    if not train_prop + valid_prop + test_prop == 1:
        raise ValueError('train_prop, valid_prop and test_prop should sum up to 1')

    if use_first_n_characters > 0:
        with open(source_file, 'r') as fp:
            text = fp.read(use_first_n_characters)
    else:
        with open(source_file, 'r') as fp:
            text = fp.read()

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    if alphabet_file is not None:
        alphabet = Alphabet.from_json(alphabet_file)
        text = ''.join(filter(lambda c: c in alphabet.char_to_int, text))

    text_len = len(text)

    if not train_prop == 0:
        with open(os.path.join(destination_dir, 'train.txt'), 'w+', encoding='utf-8') as fp:
            fp.write(text[:int(train_prop * text_len)])

    if not valid_prop == 0:
        with open(os.path.join(destination_dir, 'valid.txt'), 'w+', encoding='utf-8') as fp:
            fp.write(text[int(train_prop * text_len):int((train_prop + valid_prop) * text_len)])

    if not train_prop == 0:
        with open(os.path.join(destination_dir, 'test.txt'), 'w+', encoding='utf-8') as fp:
            fp.write(text[int((train_prop + valid_prop) * text_len):])


def update_stats_aggr(aggr, newValue):
    (count, mean, M2) = aggr
    count = count + 1
    delta = newValue - mean
    mean = mean + delta / count
    delta2 = newValue - mean
    M2 = M2 + delta * delta2
    return (count, mean, M2)
