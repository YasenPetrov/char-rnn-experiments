from collections import defaultdict
import numpy as np


class Ngram_LM:
    def __init__(self, n, alphabet):
        self.n = n
        self.alphabet = alphabet
        self.alphabet_size = alphabet.get_size()
        self.context_counts = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, train_file):
        for text in train_file.get_iterator(int(1e9)):
            for i in range(len(text) - self.n + 1):
                self.context_counts[text[i: i + self.n - 1]]['count'] += 1
                self.context_counts[text[i: i + self.n - 1]][text[i + self.n - 1]] += 1

    def predict_proba(self, context, char, delta_smoothing=0):
        if context[-self.n + 1:] in self.context_counts:
            if char in self.context_counts[context[-self.n + 1:]]:
                return (delta_smoothing + self.context_counts[context[-self.n + 1:]][char]) / \
                       (delta_smoothing * self.alphabet_size + self.context_counts[context[-self.n + 1:]]['count'])
            else:
                return delta_smoothing / \
                       (delta_smoothing * self.alphabet_size + self.context_counts[context[-self.n + 1:]]['count'])
        else:
            return 1 / self.alphabet_size

    def evaluate(self, text_file, delta_smoothing=0, adapt=False):
        log_proba_sum = 0
        total_char_count = 0
        for text in text_file.get_iterator(int(1e9)):
            # Relpace unseen characters with TOK_UNK
            text = self.alphabet.filter_string(text)

            total_char_count += len(text)
            for i in range(self.n, len(text), 1):
                prob = self.predict_proba(text[i - self.n + 1: i], text[i], delta_smoothing)
                assert (0 < prob <= 1)
                log_proba_sum += np.log2(prob)

                if adapt:
                    self.context_counts[text[i - self.n + 1: i]]['count'] += 1
                    self.context_counts[text[i - self.n + 1: i]][text[i]] += 1

        return -log_proba_sum / total_char_count

    def evaluate_string(self, text, delta_smoothing=0, adapt=False):
        # Relpace unseen characters with TOK_UNK
        text = self.alphabet.filter_string(text)

        log_proba_sum = 0
        total_char_count = 0
        total_char_count += len(text)
        for i in range(self.n, len(text), 1):
            prob = self.predict_proba(text[i - self.n + 1: i], text[i], delta_smoothing)
            log_proba_sum += np.log2(prob)

            if adapt:
                self.context_counts[text[i - self.n + 1: i]]['count'] += 1
                self.context_counts[text[i - self.n + 1: i]][text[i]] += 1

        return -log_proba_sum / total_char_count

    def get_probas(self, context, chars, delta_smoothing):
        return dict((c, self.predict_proba(context, c, delta_smoothing)) for c in chars)

    def get_distribution(self, context, delta_smoothing):
        return self.get_probas(context, self.alphabet.char_to_int.keys(), delta_smoothing)