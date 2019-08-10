import sys
import logging
import time
import csv
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from .vocab import Vocab
from .loader import Loader
from .batch import Batch
from .config import Config

from .layers import add_embedding, add_conv3f, add_softmax
from .sampler import oversample_balance


class Model(object):

    def __init__(self, config, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.logits, self.softmax = self.add_model()
        self.loss_op = self.add_loss_op(self.logits)
        self.train_op = self.add_train_op(self.loss_op)
        self.no_op = tf.no_op()
        self.logger.debug(' init ready')

    def load_data(self):
        config = self.config

        self.vocab = Vocab('Tweets', fname=config.vocab_fn)
        config.vocab_size = len(self.vocab)

        self.loader = loader = Loader(self.vocab, config.seq_len)

        if config.train:
            self.train_txt, self.train_len, self.train_label = loader.load(
                config.train_fn,
                config.max_train)

            # Balancing classes
            self.train_txt, self.train_len, self.train_label = oversample_balance(self.train_txt, self.train_len, self.train_label, verbose=True)

            self.dev_txt, self.dev_len, self.dev_label = loader.load(
                config.dev_fn,
                config.max_dev)

            self.test_txt, self.test_len, self.test_label = loader.load(
                config.test_fn,
                config.max_test)

            self.predict_txt, self.predict_len, self.predict_label = loader.load(
                config.predict_fn,
                config.max_predict)

            self.logger.info(' train(%s), dev(%s), test(%s), predict(%s) loaded.',
                             len(self.train_txt), len(self.dev_txt),
                             len(self.test_txt), len(self.predict_txt))

        self.logger.debug(' data ready')

    def add_placeholders(self):
        config = self.config

        self.text_placeholder = tf.placeholder(
            tf.int32, shape=[config.batch_size, config.seq_len],
            name='text_placeholder')

        self.label_placeholder = tf.placeholder(
            tf.int32, shape=[config.batch_size],
            name='label_placeholder')

        self.dropout_placeholder = tf.placeholder(
            tf.float32, name='dropout')

        self.logger.debug(' placeholders ready')

    def add_model(self):
        """
            @Return
                logits (tf.float32)   batch_size
        """
        config = self.config
        out = self.text_placeholder
        dp = self.dropout_placeholder

        with tf.variable_scope('Model') as scope:
            # embedding
            out = add_embedding(out,
                                config.vocab_size,
                                config.embed_size,
                                dp)

            # convolution layer with filters sizes 2, 3, 4
            # output batch_size x seq_len x (3 * conv_channels)
            out = add_conv3f('Conv3f', out, config.conv_windows,
                             config.conv_channels, dp)

            # max over time pooling
            # output batch_size x (3 * channels)
            out = tf.reduce_max(out, 1)

            # Solftmax
            logits, probs = add_softmax('Softmax', out, config.num_classes)

        self.logger.debug(' model ready')
        return logits, probs

    def add_loss_op(self, logits):
        """
            @Args
                logits (tf.float32)   batch_size
            @Return
                loss        (tf.float32)   scalar
        """
        one_hot_labels = tf.one_hot(
            indices=self.label_placeholder,
            depth=3,
            on_value=1,
            off_value=0,
            dtype=tf.int32
        )

        loss = tf.nn.softmax_cross_entropy_with_logits(
            labels=one_hot_labels, logits=logits)

        # regularization
        loss = loss + self.config.l2 * tf.reduce_sum([
            tf.nn.l2_loss(w) for w in tf.trainable_variables()])

        self.logger.debug(' loss op ready')
        return loss

    def add_train_op(self, loss):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.config.lr,
                                           beta1=self.config.adam_b1,
                                           beta2=self.config.adam_b2)
        train_op = optimizer.minimize(loss)

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
        feed_dict[self.text_placeholder] = data_txt
        # only in train mode will we have decoded batches
        if data_label is not None:
            feed_dict[self.label_placeholder] = data_label
        feed_dict[self.dropout_placeholder] = dropout
        return feed_dict

    def run_epoch(self, session, data_txt, data_len, data_label,
                  dropout=1.0, train_op=None, verbose=50):
        """Runs an epoch of training.  Trains the model for one-epoch.
        Args:
            session     (object)    tf.Session() object
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
        config = self.config

        if train_op is None:
            train_op = self.no_op

        loss_hist = []

        batch_args = (self.config.batch_size, data_txt, data_len, data_label)

        total_steps = Batch.count(*batch_args)

        batch_iter = enumerate(Batch.iterate(*batch_args))

        for step, (batch_txt, batch_len, batch_label) in batch_iter:

            feed = self.create_feed_dict(batch_txt, batch_len,
                                         batch_label, dropout)

            loss, _ = session.run([self.loss_op, train_op], feed_dict=feed)

            loss = np.mean(loss)

            loss_hist.append(loss)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r%s / %s : mean loss: %s, loss: %s      ' %
                                 (step, total_steps, np.mean(loss_hist), loss))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
            sys.stdout.flush()

        return loss_hist

    def plot_loss(self, epoch, loss_hist):
        plt.figure()
        plt.title('Training Loss')
        plt.plot(loss_hist, '-', label='Cross-Entropy Loss')
        plt.ylabel('Cross-Entropy Loss')
        plt.xlabel('Iteration')
        plt.savefig(self.config.loss_fn.replace('epoch', str(epoch)))

    def predict(self, session, data_txt, data_len, data_label=None,
                dropout=1.0, train_op=None, verbose=100, dst=None):
        """Runs an epoch of training.  Trains the model for one-epoch.
        Args:
            session     (object)    tf.Session() object
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
                loss, softmax = session.run([self.loss_op, self.softmax], feed_dict=feed)
                loss_hist.append(loss)
            else:
                _, softmax = session.run([self.no_op, self.softmax], feed_dict=feed)

            prob.append(softmax)

            if verbose and step % verbose == 0:
                sys.stdout.write('\r predicting %s / %s      ' % (step,
                                                                  total_steps))
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r done' + ' '*80)
            sys.stdout.flush()

        prob = np.concatenate(prob, axis=0)
        pred = np.argmax(prob, axis=-1)

        result_pred = np.reshape(np.array(pred), (-1, 1))

        if data_label is not None:
            correct = np.equal(data_label, pred)
            accuracy = np.sum(correct) / np.float(len(correct))
            loss = np.mean(loss_hist)
        else:
            with open(dst, 'wb') as r:
                writer = csv.writer(r)
                for result in result_pred:
                    writer.writerow(result)

        return pred, prob, accuracy, loss

    @classmethod
    def create_models(cls, config_fn):
        config = Config(config_fn)
        with tf.variable_scope('Model') as scope:
            t_model = cls(config.train)

        with tf.variable_scope('Model', reuse=True) as scope:
            p_model = cls(config.prod)

        return t_model, p_model

    @classmethod
    def train(cls, session, t_model, p_model):
        print '\n\nTRAINING\n\n'

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        best_dev_loss = float('inf')
        best_dev_epoch = 0

        session.run(init)

        for epoch in xrange(t_model.config.max_epochs):
            print 'Epoch {}'.format(epoch)
            start = time.time()

            train_loss_hist = t_model.run_epoch(
                session,
                t_model.train_txt,
                t_model.train_len,
                t_model.train_label,
                t_model.config.dropout,
                t_model.train_op,
                verbose=50)

            dev_loss_hist = t_model.run_epoch(
                session,
                t_model.dev_txt,
                t_model.dev_len,
                t_model.dev_label,
                verbose=50)

            dev_loss = np.mean(dev_loss_hist)
            train_loss = np.mean(train_loss_hist)

            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
                best_dev_train_loss = train_loss
                best_dev_epoch = epoch
                saver.save(session, t_model.config.weights_fn)

            t_model.plot_loss(epoch, train_loss_hist)

            print_args = (train_loss, dev_loss, time.time() - start)
            print 'train loss: %s\t\tdev loss: %s\t\ttime: %.3f s' % print_args

            if epoch - best_dev_epoch > t_model.config.early_stopping:
                print_args = (best_dev_epoch, best_dev_train_loss,
                              best_dev_loss, time.time() - start)
                print 'best epoch(%s): train loss: %s\t\tdev loss: %s\t\ttime: %.3f s' % print_args
                break

        # test_loss_hist = p_model.run_epoch(
        #     session,
        #     t_model.test_txt,
        #     t_model.test_len,
        #     t_model.test_label,
        #     verbose=50)
        #
        # print test_loss_hist
        #
        # print '\n=================================================\n'
        # print '*** Test Loss: %s' % np.mean(test_loss_hist)
        # print '\n=================================================\n'

        return p_model, t_model
