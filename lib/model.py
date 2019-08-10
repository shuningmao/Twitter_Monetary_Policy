import json
import logging
import tensorflow as tf
import matplotlib.pyplot as plt


class ModelConfig(object):

    def __init__(self, fp):
        with open(fp, 'rb') as f:
            self.__dict__ = json.load(f)

    def __str__(self):
        return str(self.__dict__)


class Model(object):

    def __init__(self, config, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.common = {}
        self._config = config
        self.logger.debug(' init ready')

    def customize_config(self, config):
        self.common = {}
        return config

    def build_session(self, load=False, batch_size = None):
        #if isinstance(config, str):
        #    self.config = ModelConfig(config)
        #else:
        #    self.config = config

        self.config = ModelConfig(self._config)
        self.customize_config(self.config)

        if batch_size is not None:
            self.config.batch_size = batch_size

        tf.reset_default_graph()
        self.build_nodes()
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        sess = tf.Session()
        sess.run(init_op)
        if load:
            self.load_weights(sess)
        return sess

    def build_nodes(self):
        with tf.variable_scope(self.config.model_scope):
            self.placeholder = {}
            self.add_placeholders()
            self.model_op = self.add_model_op()
            self.loss_op = self.add_loss_op(self.model_op)
            self.train_op = self.add_train_op(self.loss_op)
            self.no_op = tf.no_op()

    def save_weights(self, sess):
        self.saver.save(sess, self.config.weights_fp)
        #fp = self.config.weights_fp + self.version
        #self.saver.save(sess, fp)

    def load_weights(self, sess):
        self.saver.restore(sess, self.config.weights_fp)
        #fp = self.config.weights_fp + self.version
        #self.saver.restore(sess, fp)

    def plot_loss(self, epoch, loss_hist):
        plt.figure()
        plt.title('Training Loss')
        plt.plot(loss_hist, '-', label='Loss')
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.savefig(self.config.loss_graph_fp.format(epoch=epoch))
        plt.close()

    def load_data(self):
        raise NotImplementedError('This method must be implemented.')

    def add_placeholders(self):
        raise NotImplementedError('This method must be implemented.')

    def add_model_op(self):
        raise NotImplementedError('This method must be implemented.')

    def add_loss_op(self, model_op):
        raise NotImplementedError('This method must be implemented.')

    def add_train_op(self, loss_op):
        raise NotImplementedError('This method must be implemented.')

    def create_feed_dict(self, *args, **kwargs):
        raise NotImplementedError('This method must be implemented.')

    def run_epoch(self, *args, **kwargs):
        raise NotImplementedError('This method must be implemented.')

    def train(self, *args, **kwargs):
        raise NotImplementedError('This method must be implemented.')

    def predict(self, *args, **kwargs):
        raise NotImplementedError('This method must be implemented.')
