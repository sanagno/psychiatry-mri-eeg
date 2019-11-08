import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import os


def get_batches(iterable, batch_size=64, do_shuffle=True):
    if do_shuffle:
        iterable = shuffle(iterable)

    length = len(iterable)
    for ndx in range(0, length, batch_size):
        iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
        yield iterable_batch


def get_reconstruction_loss(true, preds, mask):
    loss = np.mean(((true - preds) ** 2) * mask, axis=1)
    return np.mean(loss, axis=0)


DEFAULT_LOG_PATH = './autoencoder'


class Autoencoder:
    training = None
    input_ = None
    input_mask = None
    intermediate_representation = None
    input_reconstructed = None
    loss = None

    def __init__(self,
                 number_of_features,
                 activation=tf.nn.relu,
                 layers=None,
                 dropout=None,
                 regularization=0,
                 masking=0.5):

        self.activation = activation

        if layers is None:
            self.layers = [15, 15, 15]
        else:
            self.layers = layers

        self.number_of_features = number_of_features

        self.masking = masking
        self.dropout = dropout

        use_regularization = (regularization > 0)
        self.use_regularization = use_regularization

        if regularization == 0:
            # set to small value to avoid tensorflow error
            # use_regularization = False in this case and will not contribute towards the final loss
            self.regularization = 0.1
        else:
            self.regularization = regularization

    def build_graph(self):

        self.training = tf.placeholder(tf.bool, shape=[], name='training')

        # these do not contain the wanted prediction
        self.input_ = tf.placeholder(tf.float32, shape=[None, self.number_of_features], name='input')

        self.input_mask = tf.placeholder(tf.float32, shape=[None, self.number_of_features], name='input_mask')

        self.intermediate_representation = self.encode(self.input_)

        self.input_reconstructed = self.decode(self.intermediate_representation)

        if self.input_mask is not None:
            self.loss = tf.reduce_mean(((self.input_ - self.input_reconstructed) ** 2) * self.input_mask)
        else:
            self.loss = tf.reduce_mean((self.input_ - self.input_reconstructed) ** 2)

        if self.use_regularization:
            self.loss += tf.losses.get_regularization_loss()

    def encode(self,
               input_):

        if self.masking > 0:
            # mask randomly some of the inputs
            input_ = tf.layers.dropout(input_, rate=self.masking, training=self.training)

        x = input_
        # important to use relu as a first layer to make all unobserved values set to 0?
        for i, layer in enumerate(self.layers):
            x = tf.layers.dense(x, layer, use_bias=True, name='input_layer_1_' + str(i),
                                activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
            if self.dropout is not None:
                x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        return x

    def decode(self,
               intermediate):

        x = intermediate
        for i, layer in enumerate(self.layers[::-1][1:]):
            x = tf.layers.dense(x, layer, use_bias=True, name='input_layer_2_' + str(i),
                                activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
            if self.dropout is not None:
                x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        x = tf.layers.dense(x, self.number_of_features, use_bias=True, name='input_layer_2_final',
                            activation=self.activation,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))

        if self.dropout is not None:
            x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        return x

    def reconstruct_with_session(self,
                                 sess,
                                 data):

        # much faster to reconstruct the whole table and make predictions from it
        data_reconstructed = np.zeros((data.shape[0], self.number_of_features))

        for rows in get_batches(list(range(data.shape[0])), batch_size=1024, do_shuffle=False):
            rows_features = [data[i, :] for i in rows]

            ratings_reconstructed = sess.run(self.input_reconstructed,
                                             feed_dict={
                                                 self.input_: rows_features,
                                                 self.training: False
                                             })

            data_reconstructed[rows] = ratings_reconstructed

        return data_reconstructed

    def reconstruct(self,
                    data,
                    log_path=None):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.build_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                reconstructed = self.reconstruct_with_session(sess, data)

                return reconstructed

    def fit(self,
            data,
            data_mask,
            test_data=None,
            test_data_mask=None,
            n_epochs=350,
            decay_steps=None,
            learning_rate=None,
            decay=None,
            log_path=None,
            verbose=True,
            print_every_epochs=10):

        if decay_steps is None:
            # empirical
            decay_steps = data.shape[0] // 64 * 5

        if learning_rate is None:
            learning_rate = 0.001

        if decay is None:
            decay = 0.96

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        validation = False
        if test_data is not None and test_data_mask is not None:
            validation = True

        with tf.Graph().as_default():
            with tf.Session() as sess:

                self.build_graph()

                global_step = tf.Variable(1, name='global_step', trainable=False)

                learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay)

                # Gradients and update operation for training the model.
                opt = tf.train.AdamOptimizer(learning_rate)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

                with tf.control_dependencies(update_ops):
                    # Update all the trainable parameters
                    train_step = opt.minimize(self.loss, global_step=global_step)

                saver = tf.train.Saver(max_to_keep=3)

                # writer = tf.summary.FileWriter(log_path, sess.graph)
                # writer.flush()

                # tf.summary.scalar('loss', self.loss)
                # tf.summary.scalar('learning_rate', learning_rate)
                # summaries_merged = tf.summary.merge_all()

                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):
                    total_loss = 0
                    for rows in get_batches(list(range(data.shape[0]))):
                        rows_features = [data[i, :] for i in rows]
                        rows_masks = [data_mask[i, :] for i in rows]

                        # summaries_merged
                        _, loss, step = sess.run([train_step, self.loss, global_step],
                                                 feed_dict={
                                                     self.input_: rows_features,
                                                     self.input_mask: rows_masks,
                                                     self.training: True
                                                 })

                        total_loss += loss * len(rows)

                    # writer.flush()
                    total_loss = total_loss / data.shape[0]

                    if epoch % print_every_epochs == 0:
                        if verbose and validation:
                            reconstructed_train = self.reconstruct_with_session(sess, data)
                            train_loss = get_reconstruction_loss(data, reconstructed_train, data_mask)

                            reconstructed_test = self.reconstruct_with_session(sess, test_data)
                            test_loss = get_reconstruction_loss(test_data, reconstructed_test, test_data_mask)

                            print('At epoch {0:3d} train loss is {1:6.4f} and reconstruction losses are in train '
                                  '{2:6.4f} and in test {3:6.4f}'.format(epoch, total_loss, train_loss, test_loss))

                        saver.save(sess, os.path.join(log_path, "model"), global_step=epoch)
