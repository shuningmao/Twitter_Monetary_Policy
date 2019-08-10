import logging
import csv
from collections import defaultdict
# import numpy as np
from .vocab_stats import VocabStats


class Vocab(object):

    def __init__(self, name='n/a', logger=None, fname=None):
        self.logger = logger or logging.getLogger(__name__)

        self.name = name
        self.word_to_index = {}
        self.index_to_word = {}
        self.word_freq = defaultdict(int)

        self.unknown = '<unk>'
        self.eos = '<eos>'

        self.add_word(self.unknown, count=0)
        self.add_word(self.eos, count=0)

        self.unknown_token = self.encode(self.unknown)
        self.eos_token = self.encode(self.eos)

        if fname:
            self.load_vocab(fname)

    def __len__(self):
        return len(self.word_freq)

    def __str__(self):
        return str(VocabStats(self))

    # def load(self, data):
    #     old_len = len(self)
    #     if self.is_full():
    #         self.logger.info('is full(%s).', old_len)
    #     else:
    #         is_full = False
    #         for line in data:
    #             line = line.strip().split(' ')
    #             for word in line:
    #                 if self.add_word(word) == self.FULL:
    #                     is_full = True
    #                     break
    #             if is_full:
    #                 break
    #         new_len = len(self)
    #         self.logger.info('loaded new(%s) total(%s).', new_len - old_len,
    #                          new_len)

    def condense(self, threshold=3):
        vocab = Vocab()
        for word, freq in self.word_freq.iteritems():
            if freq >= threshold:
                vocab.add_word(word, count=freq)
        return vocab

    #
    # def high_freq(self, threshold=None, keep_ratio=None):
    #     if threshold is None and keep_ratio is None:
    #         raise ValueError('must give threshold or keep_ratio')
    #
    #     vocab = Vocab()
    #     remaining = VocabStats()
    #     removed = VocabStats()
    #
    #     if keep_ratio is None:
    #         for word, freq in self.word_freq.iteritems():
    #             if freq > threshold:
    #                 vocab.add_word(word, count=freq)
    #                 remaining.add_word(word)
    #             else:
    #                 vocab.add_word(vocab.unknown, count=freq)
    #                 removed.add_word(word)
    #         self.logger.info('removed(<=%s) : %s', threshold, removed)
    #         self.logger.info('remaining(>%s): %s', threshold, remaining)
    #     else:
    #         remain = np.ceil((np.sum(self.word_freq.values()) - self.word_freq[self.unknown]) * keep_ratio)
    #         for word, freq in sorted(self.word_freq.iteritems(), key=lambda (k, v): -v):
    #             if remain > 0:
    #                 vocab.add_word(word, count=freq)
    #                 remaining.add_word(word)
    #                 if word != self.unknown:
    #                     remain -= freq
    #             else:
    #                 vocab.add_word(vocab.unknown, count=freq)
    #                 removed.add_word(word)
    #         self.logger.info('removed(%s) : %s', 1 - keep_ratio, removed)
    #         self.logger.info('remaining(%s): %s', keep_ratio, remaining)
    #
    #     return vocab

    def add_word(self, word, count=1):
        if not isinstance(word, str):
            raise ValueError('word must be string')

        if word not in self.word_to_index:
            index = len(self.word_to_index)
            self.word_to_index[word] = index
            self.index_to_word[index] = word

        self.word_freq[word] += count

    def encode(self, word):
        if word not in self.word_to_index:
            return self.unknown_token
        return self.word_to_index[word]

    # def decode(self, token):
    #     if token not in self.index_to_word:
    #         raise 'Cannot decode: ' + str(token)
    #     return self.index_to_word[token]

    def decode(self, token):
        if token not in self.index_to_word:
            self.logger.warn('cannot decode: %', token)
            return self.unknown
        return self.index_to_word[token]

    def encode_words(self, words):
        return [self.encode(word) for word in words]

    def decode_words(self, tokens):
        return [self.decode(token) for token in tokens]

    def _load_csv_line(self, line):
        line = line.strip().split(' ')
        for word in line:
            self.add_word(word)

    def load_csv(self, fname, col=1):
        with open(fname, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                self._load_csv_line(row[col])
        self.logger.info(' vocab loaded from %s', fname)

    def load_vocab(self, fname):
        i = -1
        with open(fname, 'rb') as f:
            reader = csv.reader(f)
            for i, row in enumerate(reader):
                self.add_word(row[0], int(row[1]))
        self.logger.info(' %s vocab loaded from %s', i + 1, fname)

    def save_vocab(self, fname):
        i = -1
        words = self.index_to_word.values()
        with open(fname, 'wb') as f:
            writer = csv.writer(f)
            for i, (word, freq) in enumerate(self.word_freq.iteritems()):
                writer.writerow([word, freq])
        self.logger.info(' %s vocab saved to %s', i + 1, fname)
