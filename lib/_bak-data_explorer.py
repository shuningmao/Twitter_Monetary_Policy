import logging
# from collections import defaultdict
import numpy as np
from lib.vocab import Vocab
import matplotlib.pyplot as plt


class DataExplorer(object):

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    def _multi_plot(self, config, export_name, xlab, ylab, plot_fn, *args):
        plt.figure()
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        for filename in config:
            plot_fn(filename, *args)
        plt.legend()
        plt.savefig(export_name)
        self.logger.info('%s exported', export_name)

    def explore_length(self, config, export_name):
        self._multi_plot(config, export_name, 'Sentence Length', 'Density',
                         self._explore_length)

    def _explore_length(self, filename):
        lengths = []
        with open(filename) as lines:
            for line in lines:
                lengths.append(len(line.split(' ')))
            self.logger.info('calculated lengths for %s (%s)',
                             filename, len(lengths))
        length_max = np.max(lengths)
        label = '%s(%s - %s)' % (filename, np.min(lengths), length_max)
        plt.plot(np.arange(length_max + 1), np.bincount(lengths) / np.float(
            len(lengths)), '-', label=label)

    def explore_change(self, config, export_name):
        self._multi_plot(config, export_name, 'Change', 'Density',
                         self._explore_change)

    def _explore_change(self, filename):
        changes = []
        with open(filename) as lines:
            for line in lines:
                changes.append(float(line.strip()))

        self.logger.info('calculated changes for %s (%s)',
                         filename, len(changes))
        x, counts = np.unique(changes, return_counts=True)
        label = '%s(%s - %s)' % (filename, np.min(changes), np.max(changes))
        plt.plot(x, counts / np.float(len(changes)), 'o', label=label)

    def explore_vocab(self, config, export_name, threshold=None, keep_ratio=None):
        vocab = Vocab()
        vocab.load_file(config[0])
        vocab = vocab.high_freq(threshold, keep_ratio).order()

        self._multi_plot(config, export_name, 'Vocab', 'Log(x+1) Max Ratio',
                         self._explore_vocab, vocab)

    def _explore_vocab(self, filename, vocab):
        words = {}
        tot = 0
        for i in xrange(1, len(vocab)):
            words[i] = 0

        with open(filename) as lines:
            for line in lines:
                for word in line.split(' '):
                    token = vocab.encode(word)
                    if token != vocab.unknown_token:
                        words[token] += 1
                        tot += 1

        keys = []
        vals = []
        for k, v in sorted(words.iteritems(), key=lambda(k, v): k):
            keys.append(k)
            vals.append(v)

        vals = np.log(np.array(vals) + 1)
        vals = vals / np.max(vals)

        self.logger.info('calculated vocabs for %s (%s)', filename, tot)
        label = '%s (%s)' % (filename, str(len(keys)))
        plt.plot(keys, vals, '.', label=label, markersize=1)

if __name__ == '__main__':
    DataExplorer().explore_files(311, [
        'data/train_txt.txt',
        'data/dev_txt.txt',
        'data/test_txt.txt'
    ])
