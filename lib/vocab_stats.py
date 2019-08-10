
import re
from collections import defaultdict


class VocabStats(object):

    def __init__(self, vocab=None):
        self.types = defaultdict(int)

        self.at = defaultdict(int)
        self.hash = defaultdict(int)
        self.word = defaultdict(int)
        self.number = defaultdict(int)
        self.char = defaultdict(int)
        self.other = defaultdict(int)

        if vocab:
            self.add_vocab(vocab)

    def __str__(self):
        return self.total_unique()

    def __len__(self):
        return sum((len(self.at), len(self.hash), len(self.word),
                    len(self.number), len(self.char), len(self.other)))


    def total_unique(self):
        return '(%s) @(%s) #(%s) word(%s) number(%s) char(%s) '\
            'other(%s)' % (len(self), len(self.at), len(self.hash),
                           len(self.word), len(self.number),
                           len(self.char), len(self.other))

    # def order(self):
    #     self.at = self.order_dict(self.at)
    #     self.hash = self.order_dict(self.hash)
    #     self.word = self.order_dict(self.word)
    #     self.number = self.order_dict(self.number)
    #     self.char = self.order_dict(self.char)
    #     self.other = self.order_dict(self.other)
    #
    # @staticmethod
    # def order_dict(input_dict):
    #     tmp = {}
    #     for word, freq in sorted(input_dict.iteritems(), key=lambda (k, v): -v):
    #         tmp[word] = freq
    #     return tmp


    def add_vocab(self, vocab):
        for word in vocab.word_freq:
            self.add_word(word)

    def add_word(self, word, count=1):
        if re.search(r'^@\w+', word):
            word_dict = self.at
        elif re.search(r'^#\w+', word):
            word_dict = self.hash
        elif re.search(r'^[A-Za-z]+$', word):
            word_dict = self.word
        elif re.search(r'^[,.0-9]*[0-9]+$', word):
            word_dict = self.number
        elif len(word) == 1:
            word_dict = self.char
        else:
            word_dict = self.other

        word_dict[word] += count
