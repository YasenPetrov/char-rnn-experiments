from collections import Counter, defaultdict
from datautils.dataset import Alphabet, TextFile
import numpy as np


class Ngram_LM:
    def __init__(self, n, alphabet_size):
        self.n = n
        self.alphabet_size = alphabet_size
        self.context_counts = defaultdict(lambda: defaultdict(lambda: 0))

    def train(self, train_file):
        for text in train_file.get_iterator(int(1e9)):
            for i in range(len(text) - self.n + 1):
                self.context_counts[text[i: i + self.n - 1]]['count'] += 1
                self.context_counts[text[i: i + self.n - 1]][text[i + self.n - 1]] += 1

    def predict_proba(self, context, char, delta_smoothing=0):
        return (delta_smoothing + self.context_counts[context[-self.n + 1:]][char]) / \
               (delta_smoothing * self.alphabet_size + self.context_counts[context[-self.n:]]['count'])

    def evaluate(self, text_file, delta_smoothing=0, adapt=False):
        log_proba_sum = 0
        total_char_count = 0
        for text in text_file.get_iterator(int(1e9)):
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