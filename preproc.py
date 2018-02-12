import argparse

from datautils.dataset import Alphabet
from utils import clean_and_split_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Preprocesses files for PPM training''')

    parser.add_argument('source_file', metavar='source_file', type=str,
                        help='A path to a the source file')
    parser.add_argument('dest_dir', metavar='dest_dir', type=str,
                        help=' path to a directory in which the processed files will be saved')
    parser.add_argument('--use_first_n', default=0, type=int, help='Use only the first n characters of the text')
    # parser.add_argument('--alphabet_file', dest='alphabet_file', action='store_const', default=False)

    args = parser.parse_args()

    clean_and_split_file(args.source_file, args.dest_dir, .9, .05, .05, use_first_n_characters=args.use_first_n
                         )