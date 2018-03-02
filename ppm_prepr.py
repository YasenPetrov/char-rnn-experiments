import os
import argparse

from datautils.dataset import Alphabet
from datautils.misc import text_file_to_ids_file


def preprocess_files_in_dir(source_dir, dest_dir):
    """
    Preprocessing for the PPM testing
    :param source_dir: A path to a directory, containing a train, vali and test .txt files
    :param dest_dir: A path to a directory in which the processed files will be saved - those will start with the
    alphabet size on the first line an then have every id on a new line - they will be called x_ids.txt for x in
    [train, valid, test]
    """
    alphabet = Alphabet.from_text(os.path.join(source_dir, 'train.txt'))

    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for fn in ['train', 'valid', 'test']:
        text_file_to_ids_file(os.path.join(source_dir, fn + '.txt'), os.path.join(dest_dir, fn + '_ids.txt'), alphabet)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='''Preprocesses files for PPM training''')

    parser.add_argument('source_dir', metavar='source_dir', type=str, help='A path to a directory, containing a train, valid and test .txt files')
    parser.add_argument('dest_dir', metavar='dest_dir', type=str, help=' path to a directory in which the processed files will be saved')

    args = parser.parse_args()

    preprocess_files_in_dir(args.source_dir, args.dest_dir)