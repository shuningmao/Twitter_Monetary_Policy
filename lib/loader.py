import logging
import csv
import numpy as np


class Loader(object):

    def __init__(self, vocab, seq_len, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.vocab = vocab
        self.seq_len = seq_len

    @staticmethod
    def len_stat(fname):
        count = []
        with open(fname, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                line = row[1]
                count.append(len(line.split(' ')))

        tmp = 'instances: %s [sequence] max:%s, min:%s, mean:%.1f, sd:%.1f'

        return tmp % (len(count), np.max(count), np.min(count),
                      np.mean(count), np.std(count))

    def load(self, fname, max_row=None, has_label=True):
        x = []
        x_len = []
        if has_label:
            y = []
        else:
            y = None
        i = -1
        with open(fname, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for i, row in enumerate(reader):
                bucket, seq_len = self.line2bucket(row[2])
                x.append(bucket)
                x_len.append(seq_len)
                if has_label:
                    y.append(int(row[3]))
                if max_row and i > max_row - 1:
                    break

        if y is not None:
            y = np.array(y)

        self.logger.info(' %s instances loaded from %s', i + 1, fname)
        return np.array(x), np.array(x_len), y

    def line2bucket(self, line):
        line = line.strip().split(' ')
        line_len = min(len(line), self.seq_len)
        result = []
        for i in xrange(line_len):
            result.append(self.vocab.encode(line[i]))
        for i in xrange(self.seq_len - line_len):
            result.append(self.vocab.eos_token)
        return result, line_len

    def bucket2line(self, line):
        result = []
        for word in line:
            if word == self.vocab.pad_token:
                break
            result.append(self.vocab.decode(word))
        return ' '.join(result)
