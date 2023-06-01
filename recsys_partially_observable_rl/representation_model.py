import tensorflow_datasets as tfds
import tensorflow as tf


class RepresentationModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, slate_size, num_topics, doc_embed_dim,
                 history_length):
        super().__init__(name="RepresentationModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._slate_size = slate_size
        self._num_topics = num_topics
        self._doc_embed_dim = doc_embed_dim
        self._history_length = history_length
        self._doc_proposal_embeddings = tf.keras.layers.Embedding(
            num_docs + 2,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_prop_embedding",
            trainable=True)
        self._doc_embeddings = tf.keras.layers.Embedding(
            num_docs + 2,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_embedding",
            trainable=True)
        self._net = tf.keras.Sequential(name="user")
        self._net.add(tf.keras.layers.SimpleRNN(32))  # rnn overfit a lot
        # self._net.add(tf.keras.layers.Dense(256))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

    @profile
    def call(self, features, training=False):
        ''' 
         'doc_id': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.int32),
         'doc_topic': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.int32),
         'doc_quality': tfds.features.Tensor(shape=(self.num_docs,), dtype=tf.float32),
         'doc_features': tfds.features.Tensor(shape=(self.num_docs, self.num_topics), dtype=tf.float32),
         'slate_doc_id': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.int32),
         'slate_doc_topic': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.int32),
         'slate_doc_quality': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size), dtype=tf.float32),
         'slate_doc_features': tfds.features.Tensor(shape=(self.history_length, 1, self.slate_size, self.num_topics), dtype=tf.float32),
         'choice': tfds.features.Tensor(shape=(self.history_length-1, 1), dtype=tf.int32),
         'consumed_time': tfds.features.Tensor(shape=(self.history_length-1, 1), dtype=tf.float32),
         'label' : tfds.features.ClassLabel(num_classes=self.num_docs+1)
        '''
        # [batch_size, history_length-1, 1]
        chosen_doc_idx = features['choice'][:, :self._history_length-1, :]
        batch_size = tf.shape(chosen_doc_idx)[0]
        # [batch_size, history_length-1, slate_size]
        slate_doc = tf.squeeze(
            features['slate_doc_id'][:, :self._history_length-1, :, :], [2])
        append_id = tf.fill(
            [batch_size, self._history_length-1, 1], self._num_docs+1)
        # [batch_size, history_length-1, slate_size+1]
        slate_doc_expanded = tf.concat([slate_doc, append_id], -1)
        doc_id_history = tf.experimental.numpy.take_along_axis(
            slate_doc_expanded, chosen_doc_idx, axis=-1)  # [batch_size, history_length-1, 1]
        doc_history_embeddings = self._doc_embeddings(
            tf.squeeze(doc_id_history, [2]))  # [batch_size, history_length-1, embed_dim]

        slate_doc_quality = tf.squeeze(
            features['slate_doc_quality'][:, :self._history_length-1, :, :], [2])
        append_quality = tf.fill(
            [batch_size, self._history_length-1, 1], 0.)
        slate_doc_quality_expanded = tf.concat(
            [slate_doc_quality, append_quality], -1)
        doc_quality_history = tf.experimental.numpy.take_along_axis(
            slate_doc_quality_expanded, chosen_doc_idx, axis=-1)  # [batch_size, history_length-1, 1]

        slate_doc_features = tf.squeeze(features['slate_doc_features'][
            :, :self._history_length-1, :, :], [2])  # [batch_size, history_length-1, slate_size, num_topics]
        append_feature = tf.zeros(
            [batch_size, self._history_length-1, 1, self._num_topics])
        # [batch_size, history_length-1, slate_size+1, num_topics]
        doc_features_expanded = tf.concat(
            [slate_doc_features, append_feature], -2)
        doc_features_history = tf.experimental.numpy.take_along_axis(
            doc_features_expanded, chosen_doc_idx[:, :, :, None], axis=2)  # [batch_size, history_length-1, 1, num_topics]

        # [batch_size, history_length-1, 1]
        c_time_history = features['consumed_time'][:,
                                                   :self._history_length-1, :]

        user_features = tf.concat(
            (doc_history_embeddings, tf.squeeze(doc_features_history, [2]), doc_quality_history, c_time_history), axis=-1)  # [batch_size, history_length-1, embed_dim + num_topics + 2]
        user_features = tf.reshape(
            user_features, [batch_size, self._num_users, -1])  # [batch_size, 1, (history_length-1)*(embed_dim + num_topics + 1)]
        user_embeddings = self._net(user_features)  # [..., embed_dim]
        # doc_features = self._doc_proposal_embeddings(
        #     tf.range(1, self._num_docs + 2, dtype=tf.int32))  # embedding, currently abandoned
        doc_features = features['doc_features'][0, :, :]
        pseudo_doc_feature = tf.zeros([1, self._num_topics])
        doc_features = tf.concat([doc_features, pseudo_doc_feature], 0)
        # scores = tf.einsum('bij,kj->bik', user_embeddings,
        #                    doc_features)  # linear version
        scores = tf.einsum('bj,kj->bk', user_embeddings,
                           doc_features)  # rnn version

        pseudo_doc = tf.fill([batch_size, 1, 1], self._num_docs+1)
        test_slate = tf.reshape(tf.concat(
            [features['slate_doc_id'][:, self._history_length-1, :], pseudo_doc], -1), [batch_size, self._slate_size+1, 1])
        test_slate = tf.subtract(test_slate, tf.cast(
            tf.ones(tf.cast(tf.shape(test_slate), tf.int32)), tf.int32))
        predict_dist = tf.experimental.numpy.take_along_axis(tf.reshape(
            scores, [batch_size, self._num_docs+1, 1]), tf.cast(test_slate, tf.int32), axis=1)  # [batch_size, slate_size+1]
        return tf.squeeze(predict_dist)


ds_split = tfds.load('recsim_dataset:3.0.0', split=[
                     'train[:50%]', 'train[98%:]'])
ds_test = ds_split[0]
ds_train = ds_split[1]
assert isinstance(ds_train, tf.data.Dataset)

batch_size = 100
num_users = 1
num_docs = 20
slate_size = 2
num_topics = 2
history_length = 20
doc_embed_dim = 2

loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)


def loss(model, features, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    y = tf.nn.softmax(tf.one_hot(features['choice'][:, -1, 0], 3))
    y_ = tf.nn.softmax(model(x, training=training))
    
    # doc_features = features['doc_features'][0, :, :]
    # pseudo_doc_feature = tf.zeros([1, model._num_topics])
    # doc_features = tf.concat([doc_features, pseudo_doc_feature], 0)
    # user_feature = features['user_state'][:, -1, 0, :]
    # scores = tf.einsum('bj,kj->bk', user_feature, doc_features)
    # y_ = tf.nn.softmax(scores)

    return loss_object(y_true=y, y_pred=y_)


def grad(model, inputs):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, training=True)
    return loss_value, tape.gradient(loss_value, model.trainable_variables)


model = RepresentationModel(num_users, num_docs, slate_size, num_topics, doc_embed_dim,
                            history_length)

# optimizer = tf.keras.optimizers.SGD(learning_rate=0.05, momentum=0.8)

train_loss_results = []
train_accuracy_results = []

num_epochs = 1

for epoch in range(num_epochs):
    # epoch_loss_avg = tf.keras.metrics.Mean()
    # epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    # ds_train = ds_train.shuffle(10000)
    # # Batch after shuffling to get unique batches at each epoch.
    # ds_train_batch = ds_train.batch(batch_size)
    # features = next(iter(ds_train_batch))

    # for features in ds_train_batch:
    #     # Optimize the model
    #     loss_value, grads = grad(model, features)
    #     # optimizer.apply_gradients(zip(grads, model.trainable_variables))

    #     # Track progress
    #     epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    #     # Compare predicted label to actual label
    #     # training=True is needed only if there are layers with different
    #     # behavior during training versus inference (e.g. Dropout).
    #     epoch_accuracy.update_state(
    #         features['choice'][:, -1, 0][:, None], model(features, training=True))

    # # End epoch
    # train_loss_results.append(epoch_loss_avg.result())
    # train_accuracy_results.append(epoch_accuracy.result())

    # # if epoch % 10 == 0:
    # print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
    #                                                             epoch_loss_avg.result(),
    #                                                             epoch_accuracy.result()))

    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    ds_test = ds_test.shuffle(250000)
    ds_test_batch = ds_test.batch(batch_size)

    for features in ds_test_batch:
        # training=False is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).

        # test_accuracy(features['choice'][:, -1, 0][:, None],
        #               model(features, training=False))

        doc_features = features['doc_features'][0, :, :]
        pseudo_doc_feature = tf.zeros([1, model._num_topics])
        doc_features = tf.concat([doc_features, pseudo_doc_feature], 0)
        user_feature = features['user_state'][:, -1, 0, :]
        # tf.print(doc_features)
        # tf.print(user_feature)
        scores = tf.einsum('bj,kj->bk', user_feature, doc_features)
        pseudo_doc = tf.fill([batch_size, 1, 1], model._num_docs+1)
        test_slate = tf.reshape(tf.concat(
            [features['slate_doc_id'][:, model._history_length-1, :], pseudo_doc], -1), [batch_size, model._slate_size+1, 1])
        test_slate = tf.subtract(test_slate, tf.cast(
            tf.ones(tf.cast(tf.shape(test_slate), tf.int32)), tf.int32))
        predict_dist = tf.experimental.numpy.take_along_axis(tf.reshape(
            scores, [batch_size, model._num_docs+1, 1]), tf.cast(test_slate, tf.int32), axis=1)  # [batch_size, slate_size+1]

        tf.print(features['choice'][:, -1, 0][:, None])
        tf.print(tf.squeeze(predict_dist))

        test_accuracy(features['choice'][:, -1, 0][:, None],
                      tf.squeeze(predict_dist))

    # if epoch % 10 == 0:
    print("Epoch {:03d}: Test set accuracy: {:.3%}".format(epoch,
                                                           test_accuracy.result()))
