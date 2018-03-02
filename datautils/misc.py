def text_file_to_ids_file(source_filename, dest_filename, alphabet, remove_unknown=False, base=0):
    """
    Reads in a file, turns the characters to IDs given an alphabet and saves those to a new file, separated by a newline
    character. The first line in the output file is the alphabet size
    :param source_filename:
    :param dest_filename:
    :param alphabet:
    :param base: Indexing starts from base
    :param remove_unknown: should characters that are not present in the alphabet be removed. If False, they are
        replaced with an UNK token

    """
    with open(source_filename, 'r', encoding='utf-8') as fp:
        text = fp.read()

    ids = alphabet.string_to_ids(text, remove_unknown)

    if not base == 0:
        for i in range(len(ids)):
            ids[i] += base

    with open(dest_filename, 'w+') as fp:
        fp.write(str(alphabet.get_size()))
        fp.write('\n')
        for id in ids:
            fp.write(str(id))
            fp.write('\n')