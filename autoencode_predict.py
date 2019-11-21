import tensorflow as tf
import numpy as np
import os
from utils import get_batches, f1_per_class, before_softmax_to_predictions, multi_label_accuracy


DEFAULT_LOG_PATH = './autoencoder_predict'


class AutoencodePredict:
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
                 alpha=1,           # parameter showing the significance of the prediction loss
                 activation=tf.nn.relu,
                 layers=None,
                 prediction_layers=None,
                 dropout=None,
                 regularization=0,
                 masking=0.5):

        self.activation = activation
        self.num_classes = num_classes
        self.alpha = alpha

        if layers is None:
            self.layers = [50, 15]
        else:
            self.layers = layers

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
        self.pos_weights = tf.placeholder(tf.float32, shape=[self.num_classes], name='pos_weights')
        self.class_weights = tf.placeholder(tf.float32, shape=[self.num_classes], name='class_weights')

        self.intermediate_representation = self.encode(self.input_)

        self.input_reconstructed = self.decode(self.intermediate_representation)

        self.predictions = self.predict_classes(self.intermediate_representation)

        if self.input_mask is not None:
            self.reconstruction_loss = tf.reduce_mean(((self.input_ - self.input_reconstructed) ** 2) * self.input_mask)
        else:
            self.reconstruction_loss = tf.reduce_mean((self.input_ - self.input_reconstructed) ** 2)

        # self.prediction_loss = tf.reduce_mean(
        #     tf.nn.sigmoid_cross_entropy_with_logits(labels=self.true_predictions,
        #                                             logits=self.predictions)) * self.alpha
        def my_loss(labels, logits, pos_weight, class_weight):
            return tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(labels=labels,
                                                                           logits=logits,
                                                                           pos_weight=pos_weight)) * class_weight

        loss_per_class = tf.map_fn(
            lambda x: my_loss(x[0], x[1], x[2], x[3]),
            (tf.transpose(self.true_predictions), tf.transpose(self.predictions), self.pos_weights, self.class_weights),
            dtype=tf.float32)

        self.prediction_loss = tf.reduce_mean(loss_per_class) * self.alpha

        if self.use_regularization:
            self.regularization_loss = tf.losses.get_regularization_loss()

            self.total_loss = self.reconstruction_loss + self.regularization_loss + self.prediction_loss
        else:
            self.total_loss = self.reconstruction_loss + self.prediction_loss

    def predict_classes(self, intermediate):

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
            test_data=None,
            test_data_mask=None,
            test_data_labels=None,
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
            decay_steps = data.shape[0] // 64 * 5

        if learning_rate is None:
            learning_rate = 0.001

        if decay is None:
            decay = 0.96

        if log_path is None:
            log_path = DEFAULT_LOG_PATH

        validation = False
        if test_data is not None and test_data_mask is not None and test_data_labels is not None:
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
                    train_step = opt.minimize(self.total_loss, global_step=global_step)

                saver = tf.train.Saver(max_to_keep=3)

                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):
                    reconstruction_loss = 0

                    if self.use_regularization:
                        regularization_loss = 0

                    prediction_loss = 0
                    for rows in get_batches(list(range(data.shape[0])), batch_size=64):
                        rows_features = data[rows]
                        rows_masks = data_mask[rows]
                        rows_predictions = data_labels[rows]

                        if self.use_regularization:
                            _, rec_loss, reg_loss, pred_loss, step = sess.run(
                                [train_step, self.reconstruction_loss, self.regularization_loss, self.prediction_loss,
                                 global_step],
                                feed_dict={
                                    self.input_: rows_features,
                                    self.input_mask: rows_masks,
                                    self.true_predictions: rows_predictions,
                                    self.pos_weights: pos_weights,
                                    self.class_weights: class_weights,
                                    self.training: True
                                })
                        else:
                            _, rec_loss, pred_loss, step = sess.run(
                                [train_step, self.reconstruction_loss, self.prediction_loss,
                                 global_step],
                                feed_dict={
                                    self.input_: rows_features,
                                    self.input_mask: rows_masks,
                                    self.true_predictions: rows_predictions,
                                    self.pos_weights: pos_weights,
                                    self.class_weights: class_weights,
                                    self.training: True
                                })

                        reconstruction_loss += rec_loss

                        if self.use_regularization:
                            regularization_loss += reg_loss

                        prediction_loss += pred_loss

                    if epoch % print_every_epochs == 0:
                        if verbose and validation:
                            predictions_train = self.predict_with_sess(sess, data)
                            train_accuracy = multi_label_accuracy(data_labels, predictions_train)

                            predictions_test = self.predict_with_sess(sess, test_data)
                            test_accuracy = multi_label_accuracy(test_data_labels, predictions_test)

                            if self.use_regularization:
                                print('At epoch {:4d} rec_loss: {:8.4f} reg_loss: {:8.4f} pred_loss: {:8.4f} train_'
                                      'acc: {:.4f} test_acc {:.4f}'.format(epoch, reconstruction_loss,
                                                                           regularization_loss, prediction_loss,
                                                                           train_accuracy, test_accuracy))
                            else:
                                print('At epoch {:4d} rec_loss: {:8.4f} pred_loss: {:8.4f} train_'
                                      'acc: {:.4f} test_acc {:.4f}'.format(epoch, reconstruction_loss,
                                                                           prediction_loss,
                                                                           train_accuracy, test_accuracy))

                            f1_scores = f1_per_class(data_labels, predictions_train)
                            print('\ttrain f1_scores: ', end='')
                            for sc in f1_scores:
                                print('{:.3f} '.format(sc), end='')

                            f1_scores = f1_per_class(test_data_labels, predictions_test)
                            print('test f1_scores: ', end='')
                            for sc in f1_scores:
                                print('{:.3f} '.format(sc), end='')
                            print()

                        saver.save(sess, os.path.join(log_path, "model"), global_step=epoch)

