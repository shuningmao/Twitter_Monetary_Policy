
import itertools, operator, random
from collections import defaultdict
import csv

def loading_file(srcs_labeled):
    content = []
    j = 0
    for src_fn in srcs_labeled:
      with open(src_fn, 'rb') as inp:
        reader = csv.reader(inp)
        content_here = list(reader)
        content_here = [[int(line[0])+j]+ line[1:] for line in content_here]
        content.append(content_here)
        j = content_here[-1][0]

    content = [line for file1 in content for line in file1]
    return content


def shuffle_file(content):


    groups = [list(group) for _, group in itertools.groupby(content, operator.itemgetter(0))]
    random.shuffle(groups)
    shuffled = [line for file in groups for line in file]
    #groups = defaultdict(list)
    #for line in itertools.izip(data, data_len, label):
        #groups[line[2]].append(line)

    return shuffled
