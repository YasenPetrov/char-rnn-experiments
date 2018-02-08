def text_file_to_ids_file(source_filename, dest_filename, alphabet):
    """
    Reads in a file, turns the characters to IDs given an alphabet and saves those to a new file, separated by a newline
    character. The first line in the output file is the alphabet size
    :param source_filename:
    :param dest_filename:
    :param alphabet:
    """
    with open(source_filename, 'r') as fp:
        text = fp.read()

    ids = alphabet.string_to_ids(text)

    with open(dest_filename, 'w+') as fp:
        fp.write(str(alphabet.get_size()))
        fp.write('\n')
        for id in ids:
            fp.write(str(id))
            fp.write('\n')