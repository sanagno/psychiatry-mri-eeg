import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score
import numpy as np
import warnings
import os


def count_correct(out1, out2, eod_token):
    # out2 is the prediction
    total_tokens = 0
    correct_tokens = 0
    for o1, o2 in zip(out1.astype(np.int32), out2.astype(np.int32)):
        for i, token in enumerate(o1):
            total_tokens += 1

            if o1[i] == o2[i]:
                correct_tokens += 1

            if o1[i] == eod_token:
                break

    return correct_tokens / total_tokens


def get_batches(iterable, batch_size=64, do_shuffle=True):
    if do_shuffle:
        iterable = shuffle(iterable)

    length = len(iterable)
    for ndx in range(0, length, batch_size):
        iterable_batch = iterable[ndx: min(ndx + batch_size, length)]
        yield iterable_batch


def get_reconstruction_loss(true, predictions, mask):
    loss = np.mean(((true - predictions) ** 2) * mask, axis=1)
    return np.mean(loss, axis=0)


def multi_label_accuracy(true, predictions):
    if not issubclass(predictions.dtype.type, np.integer):
        predictions = before_softmax_to_predictions(predictions)

    return 1 - np.sum((true - predictions) ** 2) / (true.shape[0] * true.shape[1])


def before_softmax_to_predictions(predictions):
    return (predictions >= 0).astype(np.int16)


def f1_per_class(true, predictions):
    if not issubclass(predictions.dtype.type, np.integer):
        predictions = before_softmax_to_predictions(predictions)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f1_scores = list()
        for i in range(true.shape[1]):
            f1_scores.append(f1_score(true[:, i], predictions[:, i], average='macro'))

    return f1_scores


DEFAULT_LOG_PATH = './autoencoder_predict'


# noinspection PyAttributeOutsideInit
class AutoencodeSeqPredict:
    training = None
    input_ = None
    input_mask = None
    intermediate_representation = None
    input_reconstructed = None
    reconstruction_loss = None
    regularization_loss = None
    prediction_loss = None
    true_predictions = None
    predictions = None
    total_loss = None
    pos_weights = None
    class_weights = None

    def __init__(self,
                 number_of_features,
                 num_classes,
                 vocab_size,
                 rnn_size,
                 max_generator_size,
                 embed_size,
                 bod_token,
                 eod_token,
                 pad_token,
                 alpha=1,           # parameter showing the significance of the prediction loss
                 beta=1,           # parameter showing the significance of the sequence loss
                 activation=tf.nn.relu,
                 layers=None,
                 prediction_layers=None,
                 dropout=None,
                 regularization=0,
                 masking=0.5):

        self.activation = activation
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta

        if layers is None:
            self.layers = [50, 15]
        else:
            self.layers = layers

        self.hidden_size = self.layers[-1]
        self.max_generator_size = max_generator_size
        self.vocab_size = vocab_size
        self.rnn_size = rnn_size
        self.embed_size = embed_size

        self.bod_token = bod_token
        self.eod_token = eod_token
        self.pad_token = pad_token

        if prediction_layers is None:
            self.prediction_layers = [25, 15]
        else:
            self.prediction_layers = prediction_layers

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

        self.input_ = tf.placeholder(tf.float32, shape=[None, self.number_of_features], name='input_data')
        self.input_mask = tf.placeholder(tf.float32, shape=[None, self.number_of_features], name='input_mask')
        self.true_predictions = tf.placeholder(tf.float32, shape=[None, self.num_classes], name='input_predictions')

        # placeholders used to balance the loss for individual classes and predictions
        self.pos_weights = tf.placeholder(tf.float32, shape=[5], name='pos_weights')
        self.class_weights = tf.placeholder(tf.float32, shape=[5], name='class_weights')

        self.intermediate_representation = self.encode(self.input_)

        self.input_reconstructed = self.decode(self.intermediate_representation)

        self.predictions = self.graph_predict_classes(self.intermediate_representation)

        self.sequence_graph(self.intermediate_representation)
        self.sequential_generation_graph()

        if self.input_mask is not None:
            self.reconstruction_loss = tf.reduce_mean(((self.input_ - self.input_reconstructed) ** 2) * self.input_mask)
        else:
            self.reconstruction_loss = tf.reduce_mean((self.input_ - self.input_reconstructed) ** 2)

        def my_loss(labels, logits, pos_weight, class_weight):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits,
                                                                           pos_weight=pos_weight)) * class_weight

        loss_per_class = tf.map_fn(
            lambda x: my_loss(x[0], x[1], x[2], x[3]),
            (tf.transpose(self.true_predictions), tf.transpose(self.predictions), self.pos_weights, self.class_weights),
            dtype=tf.float32)

        self.prediction_loss = tf.reduce_mean(loss_per_class) * self.alpha

        self.total_loss = self.reconstruction_loss + self.prediction_loss + self.sequence_loss

        if self.use_regularization:
            self.regularization_loss = tf.losses.get_regularization_loss()

            self.total_loss += self.regularization_loss

    def sequential_generation_graph(self):
        # sequential generation
        sequences = []  # (batch_size, max_generator_size)
        last_token = tf.tile(tf.Variable([self.bod_token], trainable=False), [tf.shape(self.proj_states)[0]])
        last_proj_states = self.proj_states

        for _ in range(self.max_generator_size - 1):
            step_rnn_states = self.gru(tf.expand_dims(tf.nn.embedding_lookup(self.embedding, last_token), 1),
                                       initial_state=last_proj_states)

            last_proj_states = tf.reshape(step_rnn_states, [-1, self.rnn_size], name='reshape_hidden')

            logits = tf.nn.relu(tf.matmul(step_rnn_states, self.W_vocab) + self.b_vocab)

            # reshape to known shape
            logits = tf.reshape(logits, [-1, self.vocab_size], name='reshape_logits')
            last_token = tf.argmax(logits, axis=1)

            sequences.append(last_token)

        self.sequences = tf.transpose(sequences)

    def sequence_graph(self, intermediate):
        self.ordered_disorders = tf.placeholder(tf.int32, [None, self.max_generator_size], name='ordered_disorders')

        self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.embed_size], -1, 1), name='embedding')
        # (batch_size, max_len, hidden_size)
        self.embedded_ordered_disorders = tf.nn.embedding_lookup(self.embedding,
                                                                 self.ordered_disorders)

        self.W_proj = tf.Variable(tf.random_uniform([self.hidden_size, self.rnn_size]))
        self.b_proj = tf.Variable(tf.zeros(shape=[self.rnn_size]))

        self.proj_states = tf.nn.relu(tf.matmul(intermediate, self.W_proj) + self.b_proj,
                                      name='vocab_projection')

        self.gru = tf.keras.layers.GRU(self.rnn_size,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

        rnn_states = self.gru(self.embedded_ordered_disorders, initial_state=self.proj_states)

        self.W_vocab = tf.Variable(tf.random_uniform([self.rnn_size, self.vocab_size]))
        self.b_vocab = tf.Variable(tf.zeros(shape=[self.vocab_size]))

        logits = tf.nn.relu(tf.matmul(rnn_states, self.W_vocab) + self.b_vocab, name='vocab_projection')

        mask = tf.cast(tf.not_equal(self.ordered_disorders[:, 1:], self.pad_token), tf.float32)

        # transforms outputs to the required shape (batch_size, sentence_length - 1, vocabulary_size)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :-1],
                                                                       labels=self.ordered_disorders[:, 1:]) * mask

        loss_per_batch_sample = tf.reduce_sum(cross_entropy, axis=1) / tf.reduce_sum(mask, axis=1)

        self.sequence_loss = tf.reduce_mean(loss_per_batch_sample) * self.beta

    def graph_predict_classes(self, intermediate):

        x = intermediate
        for i, layer in enumerate(self.prediction_layers):
            x = tf.layers.dense(x, layer, use_bias=True, name='predict_layer_' + str(i),
                                activation=self.activation,
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
            if self.dropout is not None:
                x = tf.layers.dropout(x, rate=self.dropout, training=self.training)

        x = tf.layers.dense(x, self.num_classes, use_bias=True, name='predict_layer_final', activation=None,
                            kernel_regularizer=tf.contrib.layers.l2_regularizer(self.regularization))
        return x

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

    def predict_sequences_with_sess(self, sess, data):
        predictions = np.zeros((data.shape[0], self.max_generator_size - 1))

        for rows in get_batches(list(range(data.shape[0])), batch_size=64):
            rows_features = data[rows]

            preds = sess.run(self.sequences, feed_dict={
                self.input_: rows_features,
                self.training: False
            })

            predictions[rows] = preds

        return predictions

    def predict_sequences(self,
                          data,
                          log_path=None):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.build_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                return self.predict_sequences_with_sess(sess, data)

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

                data_reconstructed = np.zeros((data.shape[0], self.number_of_features))

                for rows in get_batches(list(range(data.shape[0])), batch_size=64, do_shuffle=False):
                    rows_features = [data[i, :] for i in rows]

                    rows_reconstructed = sess.run(self.input_reconstructed,
                                                  feed_dict={
                                                      self.input_: rows_features,
                                                      self.training: False
                                                  })

                    data_reconstructed[rows] = rows_reconstructed

                return data_reconstructed

    def get_latent_space(self,
                         data,
                         log_path=None):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.build_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                data_latent = np.zeros((data.shape[0], self.layers[-1]))

                for rows in get_batches(list(range(data.shape[0])), batch_size=64, do_shuffle=False):
                    rows_features = [data[i, :] for i in rows]

                    rows_latent = sess.run(self.intermediate_representation,
                                           feed_dict={
                                               self.input_: rows_features,
                                               self.training: False
                                           })

                    data_latent[rows] = rows_latent

                return data_latent

    def predict_with_sess(self, sess, data):
        predictions = np.zeros((data.shape[0], self.num_classes))

        for rows in get_batches(list(range(data.shape[0])), batch_size=64, do_shuffle=False):
            rows_features = [data[i, :] for i in rows]

            rows_predictions = sess.run(self.predictions,
                                        feed_dict={
                                            self.input_: rows_features,
                                            self.training: False
                                        })

            predictions[rows] = rows_predictions

        return predictions

    def predict(self,
                data,
                log_path=None,
                make_integer=True):

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        with tf.Graph().as_default():
            with tf.Session() as sess:
                self.build_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path))

                if make_integer:
                    return before_softmax_to_predictions(self.predict_with_sess(sess, data))
                else:
                    return self.predict_with_sess(sess, data)

    def fit(self,
            data,
            data_mask,
            data_labels,
            data_orders,
            test_data=None,
            test_data_mask=None,
            test_data_labels=None,
            test_data_orders=None,
            pos_weights=None,
            class_weights=None,
            n_epochs=350,
            decay_steps=None,
            learning_rate=None,
            decay=None,
            log_path=None,
            verbose=True,
            print_every_epochs=10):

        if pos_weights is None:
            pos_weights = [1] * self.num_classes

        if class_weights is None:
            class_weights = [1] * self.num_classes

        if decay_steps is None:
            # empirical
            decay_steps = data.shape[0] // 64 * 20

        if learning_rate is None:
            learning_rate = 0.001

        if decay is None:
            decay = 0.96

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        validation = True
        if np.sum([1 if (x is None) else 0 for x in [test_data, test_data_mask, test_data_labels, test_data_orders]]) \
                > 0:
            validation = False

        with tf.Graph().as_default():
            with tf.Session() as sess:

                self.build_graph()

                global_step = tf.Variable(1, name='global_step', trainable=False)

                learning_rate = tf.Variable(learning_rate, trainable=False, dtype=tf.float32, name="learning_rate")
                learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay)

                # Gradients and update operation for training the model.
                # opt = tf.train.AdamOptimizer(learning_rate)
                # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                #
                # with tf.control_dependencies(update_ops):
                #     # Update all the trainable parameters
                #     train_step = opt.minimize(self.total_loss, global_step=global_step)

                optimizer = tf.train.AdamOptimizer(learning_rate)
                gvs = optimizer.compute_gradients(self.total_loss)
                capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
                train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                saver = tf.train.Saver(max_to_keep=3)

                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):
                    reconstruction_loss = 0

                    if self.use_regularization:
                        regularization_loss = 0

                    prediction_loss = 0
                    sequence_loss = 0

                    for rows in get_batches(list(range(data.shape[0])), batch_size=64):
                        rows_features = data[rows]
                        rows_masks = data_mask[rows]
                        rows_predictions = data_labels[rows]
                        rows_orders = data_orders[rows]

                        if self.use_regularization:
                            _, rec_loss, reg_loss, pred_loss, seq_loss, step = sess.run(
                                [train_step, self.reconstruction_loss, self.regularization_loss, self.prediction_loss,
                                 self.sequence_loss, global_step],
                                feed_dict={
                                    self.input_: rows_features,
                                    self.input_mask: rows_masks,
                                    self.true_predictions: rows_predictions,
                                    self.ordered_disorders: rows_orders,
                                    self.pos_weights: pos_weights,
                                    self.class_weights: class_weights,
                                    self.training: True
                                })
                        else:
                            _, rec_loss, pred_loss, seq_loss, step = sess.run(
                                [train_step, self.reconstruction_loss, self.prediction_loss, self.sequence_loss,
                                 global_step],
                                feed_dict={
                                    self.input_: rows_features,
                                    self.input_mask: rows_masks,
                                    self.true_predictions: rows_predictions,
                                    self.ordered_disorders: rows_orders,
                                    self.pos_weights: pos_weights,
                                    self.class_weights: class_weights,
                                    self.training: True
                                })

                        reconstruction_loss += rec_loss

                        if self.use_regularization:
                            regularization_loss += reg_loss

                        prediction_loss += pred_loss
                        sequence_loss += seq_loss

                    if epoch % print_every_epochs == 0:
                        if verbose and validation:
                            predictions_train = self.predict_with_sess(sess, data)
                            train_accuracy = multi_label_accuracy(data_labels, predictions_train)

                            predictions_test = self.predict_with_sess(sess, test_data)
                            test_accuracy = multi_label_accuracy(test_data_labels, predictions_test)

                            if self.use_regularization:
                                print('At epoch {:4d} rec_loss: {:8.4f} reg_loss: {:8.4f} pred_loss: {:8.4f} '
                                      'seq_loss {:8.4f} train_acc: {:.4f} test_acc {:.4f}'
                                      .format(epoch, reconstruction_loss, regularization_loss, prediction_loss,
                                              sequence_loss, train_accuracy, test_accuracy))
                            else:
                                print('At epoch {:4d} rec_loss: {:8.4f} pred_loss: {:8.4f} '
                                      'seq_loss {:8.4f} train_acc: {:.4f} test_acc {:.4f}'
                                      .format(epoch, reconstruction_loss, prediction_loss, sequence_loss,
                                              train_accuracy, test_accuracy))

                            # f1 scores based on predictions
                            f1_scores = f1_per_class(data_labels, predictions_train)
                            print('\ttrain f1_scores: ', end='')
                            for sc in f1_scores:
                                print('{:.3f} '.format(sc), end='')

                            f1_scores = f1_per_class(test_data_labels, predictions_test)
                            print('test f1_scores: ', end='')
                            for sc in f1_scores:
                                print('{:.3f} '.format(sc), end='')

                            # accuracy based on sequence
                            predictions = self.predict_with_sess(sess, data)
                            print(' seq correct tokens train {:.4f}'.format(
                                   count_correct(data_orders[:, 1:], predictions, self.eod_token)), end='')

                            # test
                            predictions = self.predict_with_sess(sess, test_data)
                            print(' and test {:.4f}'.format(
                                   count_correct(test_data_orders[:, 1:], predictions, self.eod_token)), end='')
                            print()

                        saver.save(sess, os.path.join(log_path, "model"), global_step=epoch)
