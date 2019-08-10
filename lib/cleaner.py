import logging
import re
import csv
from datetime import datetime
import numpy as np
#from .shuffle import shuffle_file, loading_file
import tools
import json

class Cleaner(object):

    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def count_lines(fname):
        i = -1
        with open(fname, 'rb') as f:
            for i, l in enumerate(f):
                pass
        return i

    def log_file(self, num, fn):
        self.logger.info(' saved %s instances to %s', num, fn)

    @staticmethod
    def data_label(src_fn):
        src_label_fn = src_fn.replace(".csv", "_label.csv")
        src_label_fn = src_fn.replace("/original/", "/label/")
        with open(src_fn, 'rb') as inp, open(src_label_fn, 'wb') as out:
            writer = csv.writer(out)
            for row in csv.reader(inp):
                if row[2] != "":
                    writer.writerow(row)
        return src_label_fn


    @staticmethod
    def predict(srcs, dst):
        predict_fn = dst.replace('<split>', 'predict')
        src_fn = srcs[0]
        with open(src_fn, 'rb') as inp, open(predict_fn, 'wb') as out:
            writer = csv.writer(out)
            header = ['GUID', 'Date', 'Content', 'Annotation', 'Ambiguous']
            writer.writerow(header)
            for row in csv.reader(inp):
                writer.writerow(row)
        return predict_fn


    def split(self, dst_inp, dst_out1, dst_out2, train=0.7, dev=0.15, test=0.15):
        srcs = tools.read_all_files(dst_inp)
        #srcs = [src_fn for src_fn in srcs if src_fn[0] == "2"]
        srcs_sorted = sorted(srcs)

        for i in xrange(3, len(srcs_sorted)):
            srcs_labeled = []
            tot_rows = 0
            for j in xrange(1, 4):
                src_fn = srcs_sorted[i-j]
                #src_label_fn = self.data_label(dst_inp + src_fn)
                #tot_rows += self.count_lines(src_label_fn)
                tot_rows += self.count_lines(dst_inp + src_fn)
                srcs_labeled.append(src_fn)

            train_num = np.int(np.ceil(train * tot_rows))
            dev_num = np.int(np.floor(dev * tot_rows))
            test_num = np.int(np.floor(test * tot_rows))

            train_fn = srcs_sorted[i].replace('.csv', '_train.csv')
            dev_fn = srcs_sorted[i].replace('.csv', '_dev.csv')
            test_fn = srcs_sorted[i].replace('.csv', '_test.csv')

            with open(dst_out1 + train_fn, 'wb') as train_f, \
                    open(dst_out1 + dev_fn, 'wb') as dev_f, \
                    open(dst_out2 + test_fn, 'wb') as test_f:

                train_writer = csv.writer(train_f)
                dev_writer = csv.writer(dev_f)
                test_writer = csv.writer(test_f)

                header = ['GUID', 'Date', 'Content', 'Annotation', 'Ambiguous']
                train_writer.writerow(header)
                dev_writer.writerow(header)
                test_writer.writerow(header)

                rows = []
                for src_fn in srcs_labeled:
                    with open(dst_inp + src_fn, 'rb') as src_f:
                        reader = csv.reader(src_f)
                        next(reader)
                        for row in reader:
                            rows.append(row)

                rows_shuffle = tools.shuffle_file(rows)
                m, num, writer = 0, train_num, train_writer
                for n in xrange(len(rows_shuffle)):
                    if m >= num:
                        if writer == train_writer:
                            self.log_file(train_num, train_fn)
                            m, num, writer = 0, dev_num, dev_writer
                        elif writer == dev_writer:
                            self.log_file(dev_num, dev_fn)
                            m, num, writer = 0, test_num, test_writer
                        else:
                            self.log_file(test_num, test_fn)
                            break
                    writer.writerow(rows_shuffle[n])
                    m += 1


    def pre_process(self, srcs, dst, start, end):
        self.start = self._parse_time(start)
        self.end = self._parse_time(end)

        num_rows = 0
        with open(dst, 'wb') as dst_file:
            self.writer = csv.writer(dst_file)
            self.writer.writerow(['Date', 'Content', 'Label'])
            for src in srcs:
                num_rows += self._add_src(src)
            self.writer = None

        self.log_file(num_rows, dst)

    @staticmethod
    def random_label(src, dst, labels=None):
        if labels is None:
            labels = [0, 1, 2]

        with open(src, 'rb') as rf, open(dst, 'wb') as df:
            reader = csv.reader(rf)
            writer = csv.writer(df)
            writer.writerow(next(reader))
            for row in reader:
                row.append(np.random.choice(labels, 1)[0])
                writer.writerow(row)

    def _add_src(self, src):
        num_rows = 0
        with open(src, 'rb') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                num_rows += self._add_row(row)
        # self.logger.info('%s rows processed from %s', num_rows, src)
        return num_rows

    def _add_row(self, row):
        row_time = datetime.strptime(row[1], '%m/%d/%y %H:%M')
        if self.start <= row_time and row_time < self.end:
            self.writer.writerow([row[1], self._clean_txt(row)])
            return 1
        return 0

    @staticmethod
    def _parse_time(time):
        return datetime.strptime(time, '%y%m%d %H:%M')

    @staticmethod
    def _clean_txt(row):
        content = row[3].strip().lower()
        # remove links/emails
        content = re.sub(r'(\w+://[^\s]+)|([\w.-]+@[\w.-]+)', '', content)
        # remove extra whitespaces
        content = re.sub(r'\s+', ' ', content)
        # shrink ?!,...
        content = re.sub(r'\!{2,}', '!', content)
        content = re.sub(r'\?{2,}', '?', content)
        content = re.sub(r'[\!\?]{2,}', '!?', content)
        content = re.sub(r'\,{2,}', ',', content)
        content = re.sub(r'\.{2,}', '...', content)
        # space around @/#/numbers/dates/words/special chars
        content = re.sub(r'([@#]\w+|[\d.,/]*\d+|[a-z]+|\.{3}|\!\?'
                         r'|[\"\'~`!$%&*?.,,])', r' \g<1> ', content)

        # remove extra whitespaces again
        content = re.sub(r'\s+', ' ', content)
        return content

    @staticmethod
    def json_file(base_file, dst_inp, dst_out, version):
        # srcs would be the FOMC meeting dates: e.g. 2015-06
        srcs = tools.read_all_files(dst_inp)
        srcs_sorted = sorted(srcs)
        srcs = srcs_sorted[3:]
        with open(base_file, 'r') as f1:
            base = json.load(f1)

        for src in srcs:
            base_temp = base
            for key, value in base_temp[u'train'].iteritems():
                if isinstance(value, basestring) and "<file>" in value:
                        base_temp[u'train'][key] = value.replace("<file>", src[:-4])
                if isinstance(value, basestring) and "<version>" in value:
                        base_temp[u'train'][key] = value.replace("<version>", version)
            fn = src.replace(".csv", "_config.json")
            with open(dst_out + fn, 'w') as f:
                json.dump(base_temp, f)


    @staticmethod
    def json_re_file(base_file, dst_inp, dst_out, version):
        # srcs would be the FOMC meeting dates: e.g. 2015-06
        srcs = tools.read_all_files(dst_inp)
        srcs_sorted = sorted(srcs)
        srcs = srcs_sorted[3:]
        with open(base_file, 'r') as f1:
            base = json.load(f1)

        for src in srcs:
            base_temp = dict(base)
            for key, value in base_temp.iteritems():
                if isinstance(value, basestring) and "<file>" in value:
                        base_temp[key] = value.replace("<file>", src[:-4])
                if isinstance(value, basestring) and "<version>" in value:
                        base_temp[key] = base_temp[key].replace("<version>", version)
            fn = src.replace(".csv", "_" + version + "_config.json")
            with open(dst_out + fn, 'w') as f:
                json.dump(base_temp, f)
