def count_correct(out1, out2):
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

log_path_disorder_generator = './gru'

class DisorderGenerator:
    def __init__(self, hidden_size, embed_size, rnn_size, vocab_size, max_generator_size):
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.rnn_size = rnn_size
        self.vocab_size = vocab_size
        self.max_generator_size = max_generator_size

    def build_graph(self):
        self.ordered_disorders = tf.placeholder(tf.int32, [None, self.max_generator_size], name='ordered_disorders')
        self.initial_states = tf.placeholder(tf.float32, [None, self.hidden_size], name='initial_states')

        self.embedding = tf.Variable(tf.random_uniform([self.vocab_size, self.hidden_size], -1, 1), name='embedding')
        self.embeded_ordered_disorders = tf.nn.embedding_lookup(self.embedding, self.ordered_disorders) # (batch_size, max_len, hidden_size)

        self.W_proj = tf.Variable(tf.random_uniform([self.hidden_size, self.rnn_size]))
        self.b_proj = tf.Variable(tf.zeros(shape=[self.rnn_size]))

        self.proj_states = tf.nn.relu(tf.matmul(self.initial_states, self.W_proj) + self.b_proj, name='vocab_projection')

        self.gru = tf.keras.layers.GRU(self.rnn_size,
                                       return_sequences=True,
                                       recurrent_initializer='glorot_uniform')

        rnn_states = self.gru(self.embeded_ordered_disorders, initial_state=self.proj_states)

        self.W_vocab = tf.Variable(tf.random_uniform([rnn_size, vocab_size]))
        self.b_vocab = tf.Variable(tf.zeros(shape=[vocab_size]))

        logits = tf.nn.relu(tf.matmul(rnn_states, self.W_vocab) + self.b_vocab, name='vocab_projection')

        mask = tf.cast(tf.not_equal(self.ordered_disorders[:, 1:], pad_token), tf.float32)

        # transforms outputs to the required shape (batch_size, sentence_length - 1, vocabulary_size)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits[:, :-1],
                                                                    labels=self.ordered_disorders[:, 1:]) * mask


        loss_per_batch_sample = tf.reduce_sum(cross_entropy, axis=1) / tf.reduce_sum(mask, axis=1)

        self.loss = tf.reduce_mean(loss_per_batch_sample)

    def sequential_generation_graph(self):
        # sequential generation

        sequences = [] # (batch_size, max_generator_size)
        last_token = tf.tile(tf.Variable([bod_token], trainable=False), [tf.shape(self.proj_states)[0]])
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

    def predict_with_sess(self, sess, data):
        predictions = np.zeros((data.shape[0], self.max_generator_size - 1))

        for rows in get_batches(list(range(data.shape[0])), batch_size=64):
            rows_features = data[rows]

            preds = sess.run(self.sequences, feed_dict={self.initial_states: rows_features})
            predictions[rows] = preds

        return predictions

    def predict(self, data):
        with tf.Graph().as_default():
            with tf.Session() as sess:

                self.build_graph()
                self.sequential_generation_graph()

                saver = tf.train.Saver()
                saver.restore(sess, tf.train.latest_checkpoint(log_path_disorder_generator))

                return self.predict_with_sess(sess, data)

    def fit(self, data, data_labels, test_data, test_data_labels, n_epochs=500, print_every_epoch=10):
        with tf.Graph().as_default():
            with tf.Session() as sess:

                self.build_graph()
                self.sequential_generation_graph()

                global_step = tf.Variable(1, name='global_step', trainable=False)

                optimizer = tf.train.AdamOptimizer()
                gvs = optimizer.compute_gradients(self.loss)
                capped_gvs = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gvs]
                train_step = optimizer.apply_gradients(capped_gvs, global_step=global_step)

                saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)
                sess.run(tf.global_variables_initializer())

                for epoch in range(n_epochs):
                    total_loss = 0
                    
                    for rows in get_batches(list(range(data.shape[0])), batch_size=64):
                        rows_features = data[rows]
                        rows_predictions = data_labels[rows]

                        _, loss_, step = sess.run(
                            [train_step, self.loss, global_step],
                            feed_dict={
                                self.initial_states: rows_features,
                                self.ordered_disorders: rows_predictions
                            })

                        total_loss += loss_

                    if epoch % print_every_epoch == 0:
                        print('At epoch {:4d} loss is {:8.4f}. '.format(epoch, total_loss), end='')
                        # train
                        predictions = self.predict_with_sess(sess, data)
                        print('Total correct tokens train {:.4f}'.format(count_correct(train_orders[:, 1:], predictions)), end='')

                        # test
                        predictions = self.predict_with_sess(sess, test_data)
                        print(' and test {:.4f}'.format(count_correct(test_orders[:, 1:], predictions)))

                        saver.save(sess, os.path.join(log_path_disorder_generator, "model"), global_step=epoch)