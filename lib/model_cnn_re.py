import sys
import time
import csv
import tensorflow as tf
import numpy as np

from .model import Model
from .vocab import Vocab
from .loader import Loader
from .batch import Batch

from .layers import add_embedding, add_conv3f, add_softmax
from .sampler import oversample_balance


class CNNModel(Model):

    def customize_config(self, config):
        vocab, loader = self.load_vocab()
        self.common['vocab'] = vocab
        self.common['loader'] = loader
        self.config.vocab_size = len(vocab)
        return config

    def load_vocab(self):
        vocab = Vocab('vocab', fname=self.config.vocab_fp)
        loader = Loader(vocab, self.config.seq_len)
        return vocab, loader

    def load_data(self):
        loader = self.common['loader']

        train_txt, train_len, train_label = loader.load(
            self.config.train_fp,
            self.config.max_train
        )

        # Balancing classes
        train_txt, train_len, train_label = oversample_balance(
            train_txt,
            train_len,
            train_label,
            verbose=True
        )

        dev_txt, dev_len, dev_label = loader.load(
            self.config.dev_fp,
            self.config.max_dev
        )

        dev_txt, dev_len, dev_label = oversample_balance(
            dev_txt,
            dev_len,
            dev_label,
            verbose=True
        )

        test_txt, test_len, test_label = loader.load(
            self.config.test_fp,
            self.config.max_test
        )

        self.logger.info(
            ' train(%s), dev(%s), test(%s).',
            len(train_txt), len(dev_txt), len(test_txt)
        )

        self.logger.debug(' data ready')
        return {
            'train_txt': train_txt,
            'train_len': train_len,
            'train_label': train_label,
            'dev_txt': dev_txt,
            'dev_len': dev_len,
            'dev_label': dev_label,
            'test_txt': test_txt,
            'test_len': test_len,
            'test_label': test_label
        }

    def add_placeholders(self):
        self.placeholder['text'] = tf.placeholder(
            tf.int32,
            shape=[self.config.batch_size, self.config.seq_len],
            name='placeholder_text'
        )

        self.placeholder['label'] = tf.placeholder(
            tf.int32,
            shape=[self.config.batch_size],
            name='placeholder_label'
        )

        self.placeholder['dropout'] = tf.placeholder(
            tf.float32,
            name='placeholder_dropout'
        )

        self.logger.debug(' placeholders ready')

    def add_model_op(self):
        """
            @Return
                probs (tf.float32)   batch_size
        """
        config = self.config
        dropout = self.placeholder['dropout']

        with tf.variable_scope('model'):
            # embedding
            embeded_txt = add_embedding(
                self.placeholder['text'],
                config.vocab_size,
                config.embed_size,
                dropout
            )

            # convolution layer with filters sizes 2, 3, 4
            # output batch_size x seq_len x (3 * conv_channels)
            layer1 = add_conv3f('conv3f', embeded_txt, config.conv_windows,
                                config.conv_channels, dropout)

            # max over time pooling
            # output batch_size x (3 * channels)
            layer2 = tf.reduce_max(layer1, 1)

            # Softmax
            _, probs = add_softmax('softmax', layer2, config.num_classes)

        self.logger.debug(' model op ready')
        return probs

    def add_loss_op(self, model_op):
        """
            @Args
                logits (tf.float32)   batch_size
            @Return
                loss   (tf.float32)   scalar
        """
        one_hot_labels = tf.one_hot(
            indices=self.placeholder['label'],
            depth=3,
            # 1: rate hike, 2: no rate hike, 0: no info
            on_value=1,
            off_value=0,
            dtype=tf.int32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot_labels, logits=model_op
        )

        # regularization
        loss = loss + self.config.l2 * tf.reduce_sum([
            tf.nn.l2_loss(w) for w in tf.trainable_variables()
        ])

        self.logger.debug(' loss op ready')
        return loss

    def add_train_op(self, loss_op):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config.lr,
            beta1=self.config.adam_b1,
            beta2=self.config.adam_b2
        )
        train_op = optimizer.minimize(loss_op)

        self.logger.debug(' train op ready')
        return train_op

    def create_feed_dict(self, data_txt, data_len, data_label=None,
                         dropout=1.0):
        """
            @Args
                data_txt      (np.float32)      batch_size x seq_len
                data_len    (np.int32)        batch_size
                data_label    (np.float32)      batch_size
                dropout        (float)         scalar
                return         (dictionary)    feed_dict
        """
        feed_dict = {}
        feed_dict[self.placeholder['text']] = data_txt
        # we have decoded batches only in train/dev/test mode
        if data_label is not None:
            feed_dict[self.placeholder['label']] = data_label
        feed_dict[self.placeholder['dropout']] = dropout
        return feed_dict

    def run_epoch(self, sess, data_txt, data_len, data_label,
                  dropout=1.0, train_op=None, verbose=50):
        """Runs an epoch of train/dev/test/pred.  Trains the model for one-epoch.
        Args:
            sess     (object)    tf.Session() object
            data_txt   (int)       sample_size x seq_len
            data_len (int)       sample_size
            data_label (float)     sample_size
            dropout     (float)     scalar
            train_op    (object)    train_op
            verbose     (int)       log info on every verbose number of batches
        Returns:
            mean_loss:  (float)     scalar, epoch loss
            loss_hist:  (float)     num_batches, epoch loss history
        """
        # config = self.config

        if train_op is None:
            train_op = self.no_op

        loss_hist = []

        batch_args = (self.config.batch_size, data_txt, data_len, data_label)

        total_steps = Batch.count(*batch_args)

        batch_iter = enumerate(Batch.iterate(*batch_args))

        for step, (batch_txt, batch_len, batch_label) in batch_iter:

            feed = self.create_feed_dict(batch_txt, batch_len,
                                         batch_label, dropout)

            loss, _ = sess.run([self.loss_op, train_op], feed_dict=feed)

            loss = np.mean(loss)

            loss_hist.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r%s / %s : mean epoch loss: %s, mean batch loss: %s      ' %
                                 (step, total_steps, np.mean(loss_hist), loss))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        return loss_hist

    def train_epoch(self, sess, epoch, data):
        print 'Epoch {}'.format(epoch)
        start = time.time()

        train_loss_hist = self.run_epoch(
            sess,
            data['train_txt'],
            data['train_len'],
            data['train_label'],
            dropout=self.config.dropout,
            train_op=self.train_op,
            verbose=5)

        dev_loss_hist = self.run_epoch(
            sess,
            data['dev_txt'],
            data['dev_len'],
            data['dev_label'],
            verbose=5)

        train_loss = np.mean(train_loss_hist)
        dev_loss = np.mean(dev_loss_hist)

        self.plot_loss(epoch, train_loss_hist)

        print_args = (train_loss, dev_loss, time.time() - start)
        #print 'train loss: %s\t\tdev loss: %s\t\ttime: %.3f s' % print_args
        #print 'train loss: {train_loss:.f5}\t\tdev loss: {dev_loss:.f5}\t\t time: {time_used:.3f }s '.format(
        #    train_loss=train_loss,
        #    dev_loss=dev_loss,
        #    time_used=time.time() - start
        #)

        return train_loss, dev_loss

    def train(self):
        print '\n\nTRAINING\n\n'

        sess = self.build_session(load=False)
        data = self.load_data()
        best_dev_loss = float('inf')
        best_dev_epoch = 0

        for epoch in xrange(self.config.max_epochs):
            train_loss, dev_loss = self.train_epoch(sess, epoch, data)

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_train_loss = train_loss
                best_dev_epoch = epoch
                self.save_weights(sess)

            #if epoch - best_dev_epoch > self.config.early_stopping:
            #    print 'best epoch({best_epoch:.f5}): train loss: {train_loss:.f5}\t\tdev loss: {dev_loss:.f5}'.format(
            #        best_epoch=best_dev_epoch,
            #        train_loss=best_dev_train_loss,
            #        dev_loss=best_dev_loss
            #    )
            #    break

            if epoch - best_dev_epoch > self.config.early_stopping:
                print_args = (best_dev_epoch, best_dev_train_loss,
                              best_dev_loss)
                print 'best epoch(%s): train loss: %s\t\tdev loss: %s' % print_args
                break

        test_loss_hist = self.run_epoch(
            sess,
            data['test_txt'],
            data['test_len'],
            data['test_label'],
            verbose=5)

        print '\n=================================================\n'
        print '*** Test Loss: %s' % np.mean(test_loss_hist)
        print '\n=================================================\n'

    def predict(self, data_fp, dst_fp=None, verbose=100, has_label_here = False):
        """Runs an epoch of training.  Trains the model for one-epoch.
        Args:
            sess     (object)    tf.Session() object
            data_txt   (int)       sample_size x seq_len
            data_len (int)       sample_size
            data_label (int)     sample_size
            dropout     (float)     scalar
            train_op    (object)    train_op
            verbose     (int)       log info on every verbose number of batches
        Returns:
            mean_loss:  (float)     scalar, epoch loss
            loss_hist:  (float)     num_batches, epoch loss history
            r_sq:       (float)     R squared
        """
        print '\n\nPREDICTING\n\n'

        sess = self.build_session(load=True, batch_size = 1)
        loader = self.common['loader']
        data_txt, data_len, data_label = loader.load(data_fp, has_label = has_label_here)
        #data_txt, data_len, data_label = loader.load(data_fp)
        pred = []
        prob = []
        accuracy = None
        loss_hist = []
        loss = None

        batch_args = (self.config.batch_size, data_txt, data_len, data_label)

        total_steps = Batch.count(*batch_args)

        batch_iter = enumerate(Batch.iterate(*batch_args))

        for step, (batch_txt, batch_len, batch_label) in batch_iter:
            feed = self.create_feed_dict(batch_txt, batch_len, batch_label)
            if batch_label:
                softmax, loss = sess.run(
                    [self.model_op, self.loss_op], feed_dict=feed)
                loss_hist.append(loss)
            else:
                softmax, loss = sess.run(
                    [self.model_op, self.no_op], feed_dict=feed)

            prob.append(softmax)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r predicting %s / %s      ' % (step,
                                                                  total_steps))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r done' + ' ' * 80)
            sys.stdout.flush()

        prob = np.concatenate(prob, axis=0)
        pred = np.argmax(prob, axis=-1)

        #result_pred = np.reshape(np.array(pred), (-1, 1))

        if data_label is not None:
            correct = np.equal(data_label, pred)
            accuracy = np.sum(correct) / np.float(len(correct))
            loss = np.mean(loss_hist)
        else:
            print "no data_label"

        if dst_fp:
            dst_fp = dst_fp.replace(".csv", "_" + self.config.version + ".csv")
            with open(dst_fp, 'w') as r:
                writer = csv.writer(r)
                header = ["GUID", "Date", "Content", "ML_Label_"+ self.config.version]
                writer.writerow(header)
                with open(data_fp, 'rb') as inp:
                    next(inp)
                    reader = csv.reader(inp)
                    content = list(reader)
                    result_pred = list(pred)
                    content_here = []
                    for i in xrange(len(pred)):
                        line = content[i][0:3] + [result_pred[i]]
                        content_here.append(line)
                    for line in content_here:
                        writer.writerow(line)
