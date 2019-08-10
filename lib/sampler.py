
import itertools
from collections import defaultdict
import numpy as np


def group_classes(data, data_len, label):
    groups = defaultdict(list)
    for line in itertools.izip(data, data_len, label):
        groups[line[2]].append(line)
    return groups


def oversample_balance(data, data_len, label, verbose=False):
    original_len = len(data)
    groups = group_classes(data, data_len, label)

    max_count = np.max([len(v) for l, v in groups.iteritems()])

    data_pairs = []
    for l, v in groups.iteritems():
        v = np.array(v)
        data_pairs.append(v)
        print l, len(v)
        if len(v) < max_count:
            idx = np.random.choice(np.arange(len(v)), max_count - len(v))
            data_pairs.append(v[idx])

    data_pairs = np.concatenate(data_pairs)

    np.random.shuffle(data_pairs)

    data, data_len, label = zip(*data_pairs)

    if verbose:
        print 'oversample from %s to %s' % (original_len, len(data))

    return list(data), list(data_len), list(label)
