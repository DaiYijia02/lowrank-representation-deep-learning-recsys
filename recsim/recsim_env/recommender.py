# coding=utf-8
# Copyright 2022 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Recommendation agents."""
import gin
from gym_env.gym import spaces
import numpy as np
from recsim_ng.core import value
import recsim_env.selector_lib as selector_lib
from recsim_ng.entities.recommendation import recommender
from recsim_ng.entities.state_models import dynamic
from recsim_ng.entities.state_models import estimation
from recsim_ng.lib.tensorflow import field_spec
import tensorflow as tf
import tensorflow_probability as tfp
import random

tfd = tfp.distributions
Value = value.Value
ValueSpec = value.ValueSpec
Space = field_spec.Space


class CollabFilteringModel(tf.keras.Model):
    """A tf.keras model that returns score for each (user, document) pair."""

    def __init__(self, num_users, num_docs, doc_embed_dim,
                 history_length):
        super().__init__(name="CollabFilteringModel")
        self._num_users = num_users
        self._history_length = history_length
        self._num_docs = num_docs
        self._doc_embed_dim = doc_embed_dim
        self._doc_proposal_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_prop_embedding")
        self._doc_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_embedding")
        self._net = tf.keras.Sequential(name="recs")
        self._net.add(tf.keras.layers.Dense(10))
        self._net.add(tf.keras.layers.LeakyReLU())
        self._net.add(
            tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

    def call(self, doc_id_history,
             c_time_history):
        # Map doc id to embedding.
        # [num_users, history_length, embed_dim]
        doc_history_embeddings = self._doc_embeddings(doc_id_history)
        # Append consumed time to representation.
        # [num_users, history_length, embed_dim + 1]
        user_features = tf.concat(
            (doc_history_embeddings, c_time_history[Ellipsis, np.newaxis]), axis=-1)
        # Flatten and run through network to encode history.
        user_features = tf.reshape(user_features, (self._num_users, -1))
        user_embeddings = self._net(user_features)
        # Score is an inner product between the proposal embeddings and the encoded
        # history.
        # [num_docs, embed_dim + 1]
        doc_features = self._doc_proposal_embeddings(
            tf.range(1, self._num_docs + 1, dtype=tf.int32))
        scores = tf.einsum("ik, jk->ij", user_embeddings, doc_features)
        return scores


class RepRecModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, num_topics, doc_embed_dim,
                 history_length, slate_size):
        super().__init__(name="RepresentationModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._num_topics = num_topics
        self._doc_embed_dim = doc_embed_dim
        self._history_length = history_length
        self._slate_size = slate_size
        self._doc_proposal_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_prop_embedding",
            trainable=True)
        self._doc_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_embedding",
            trainable=True)
        self._net = tf.keras.Sequential(name="user")
        self._net.add(tf.keras.layers.SimpleRNN(16))  # rnn overfit a lot
        # self._net.add(tf.keras.layers.Dense(256))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

    def call(self, slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history, available_documents):
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
        if available_documents == None:
            return tf.zeros([self._num_docs])
        
        doc_history_embeddings = self._doc_embeddings(docid_history)

        append_quality = tf.fill(
            [self._num_users, self._history_length, 1], 0.)
        slate_doc_quality_expanded = tf.concat(
            [slate_doc_quality_history, append_quality], -1)
        # doc_quality_history = tf.experimental.numpy.take_along_axis(
        #     slate_doc_quality_expanded, chosen_doc_idx, axis=-1)

        append_feature = tf.zeros(
            [self._num_users, self._history_length, 1, self._num_topics])
        slate_doc_features_expanded = tf.concat(
            [slate_doc_features_history, append_feature], -2)
        slate_doc_features_flat = tf.reshape(slate_doc_features_expanded, [self._num_users, self._history_length,-1])
        # doc_features_history = tf.experimental.numpy.take_along_axis(
        #     doc_features_expanded, chosen_doc_idx[:, :, :, None], axis=2)

        ctime_history_expanded = tf.expand_dims(ctime_history, -1)

        user_features = tf.concat(
            (doc_history_embeddings, slate_doc_quality_expanded, slate_doc_features_flat, ctime_history_expanded), axis=-1)  # [batch_size, history_length-1, embed_dim + num_topics + 2]
        # user_features = tf.reshape(
        #     user_features, [batch_size, self._num_users, -1])  # [batch_size, 1, (history_length-1)*(embed_dim + num_topics + 1)]
        user_embeddings = self._net(user_features)  # [..., embed_dim]
        # doc_features = self._doc_proposal_embeddings(
        #     tf.range(1, self._num_docs + 2, dtype=tf.int32))  # embedding, currently abandoned
        doc_features = available_documents.map(lambda field: tf.gather(field, range(self._num_docs))).get('doc_features')
        pseudo_doc_feature = tf.zeros([1, self._num_topics])
        doc_features = tf.concat([pseudo_doc_feature, doc_features], 0)
        # scores = tf.einsum('bij,kj->bik', user_embeddings,
        #                    doc_features)  # linear version
        scores = tf.einsum('bj,kj->bk', user_embeddings,
                           doc_features)  # rnn version
        return scores
    
class RepresentationModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, num_topics, doc_embed_dim,
                 history_length, slate_size):
        super().__init__(name="RepresentationModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._num_topics = num_topics
        self._doc_embed_dim = doc_embed_dim
        self._history_length = history_length
        self._slate_size = slate_size
        self._doc_proposal_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_prop_embedding",
            trainable=True)
        self._doc_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_embedding",
            trainable=True)
        self._net = tf.keras.Sequential(name="user")
        self._net.add(tf.keras.layers.SimpleRNN(16))  # rnn overfit a lot
        # self._net.add(tf.keras.layers.Dense(256))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._doc_embed_dim, name="hist_emb_layer"))

    def call(self, slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history, available_documents):
        if available_documents == None:
            return tf.zeros([self._num_docs])
        
        doc_history_embeddings = self._doc_embeddings(docid_history)

        append_quality = tf.fill(
            [self._num_users, self._history_length, 1], 0.)
        slate_doc_quality_expanded = tf.concat(
            [slate_doc_quality_history, append_quality], -1)
        # doc_quality_history = tf.experimental.numpy.take_along_axis(
        #     slate_doc_quality_expanded, chosen_doc_idx, axis=-1)

        append_feature = tf.zeros(
            [self._num_users, self._history_length, 1, self._num_topics])
        slate_doc_features_expanded = tf.concat(
            [slate_doc_features_history, append_feature], -2)
        slate_doc_features_flat = tf.reshape(slate_doc_features_expanded, [self._num_users, self._history_length,-1])
        # doc_features_history = tf.experimental.numpy.take_along_axis(
        #     doc_features_expanded, chosen_doc_idx[:, :, :, None], axis=2)

        ctime_history_expanded = tf.expand_dims(ctime_history, -1)

        user_features = tf.concat(
            (doc_history_embeddings, slate_doc_quality_expanded, slate_doc_features_flat, ctime_history_expanded), axis=-1)  # [batch_size, history_length-1, embed_dim + num_topics + 2]
        # user_features = tf.reshape(
        #     user_features, [batch_size, self._num_users, -1])  # [batch_size, 1, (history_length-1)*(embed_dim + num_topics + 1)]
        user_embeddings = self._net(user_features)  # [..., embed_dim]
        # doc_features = self._doc_proposal_embeddings(
        #     tf.range(1, self._num_docs + 2, dtype=tf.int32))  # embedding, currently abandoned
        doc_features = available_documents.map(lambda field: tf.gather(field, range(self._num_docs))).get('doc_features')
        pseudo_doc_feature = tf.zeros([1, self._num_topics])
        doc_features = tf.concat([pseudo_doc_feature, doc_features], 0)
        # scores = tf.einsum('bij,kj->bik', user_embeddings,
        #                    doc_features)  # linear version
        scores = tf.einsum('bj,kj->bk', user_embeddings,
                           doc_features)  # rnn version
        return scores
    

class RepPhiModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, num_topics, user_embed_dim, doc_embed_dim,
                 history_length, slate_size):
        super().__init__(name="RepPhiModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._num_topics = num_topics
        self._user_embed_dim = user_embed_dim
        self._doc_embed_dim = doc_embed_dim
        self._history_length = history_length
        self._slate_size = slate_size
        self._doc_embeddings = tf.keras.layers.Embedding(
            num_docs + 1,
            doc_embed_dim,
            embeddings_initializer=tf.compat.v1.truncated_normal_initializer(),
            mask_zero=True,
            name="doc_embedding",
            trainable=True)
        self._net = tf.keras.Sequential(name="phi")
        self._net.add(tf.keras.layers.SimpleRNN(16,name="phi1"))  # rnn overfit a lot
        # self._net.add(tf.keras.layers.Dense(256))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._user_embed_dim,name="phi2"))

    def call(self, slate_doc_quality_history, slate_doc_features_history, choice_history, ctime_history): 
        append_quality = tf.fill(
            [self._num_users, self._history_length, 1], 0.)
        slate_doc_quality_expanded = tf.concat(
            [slate_doc_quality_history, append_quality], -1)

        append_feature = tf.zeros(
            [self._num_users, self._history_length, 1, self._num_topics])
        slate_doc_features_expanded = tf.concat(
            [slate_doc_features_history, append_feature], -2)
        slate_doc_features_flat = tf.reshape(slate_doc_features_expanded, [self._num_users, self._history_length,-1])

        choice_history_expanded = tf.expand_dims(choice_history, -1)

        ctime_history_expanded = tf.expand_dims(ctime_history, -1)

        user_features = tf.concat(
            (slate_doc_quality_expanded, slate_doc_features_flat, tf.cast(choice_history_expanded, dtype=tf.float32), ctime_history_expanded), axis=-1)  # [batch_size, history_length-1, embed_dim + num_topics + 2]
        user_embeddings = self._net(user_features)  # [..., embed_dim]
        return user_embeddings

class RepMuModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, user_embed, num_topics, slate_size):
        super().__init__(name="RepMuModel")
        self._num_users = num_users
        self._user_embed = user_embed
        self._num_topics = num_topics
        self._slate_size = slate_size
        self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
            1, (self._num_users,),
            -np.Inf * tf.ones(self._num_users))
        self._net = tf.keras.Sequential(name="mu")
        # self._net.add(tf.keras.layers.SimpleRNN(16))  # rnn overfit a lot
        self._net.add(tf.keras.layers.Dense(256,name="mu1"))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._slate_size,name="mu2"))

    def call(self, user_vec, slate_docs_quality, slate_docs_features):
        if slate_docs_quality==None:
            scores = self._net(tf.zeros([self._num_users,self._user_embed+self._slate_size*(1+self._num_topics)]))
            doc_indice = self._document_sampler.choice(scores).get("choice")
            return doc_indice
        # slate_docs_quality = slate_docs.get("doc_quality")
        # slate_docs_features = slate_docs.get("doc_features")
        input_features = tf.concat([user_vec, tf.reshape(tf.expand_dims(slate_docs_quality,-1),[1,-1]), tf.reshape(slate_docs_features,[1,-1])],-1)
        scores = self._net(input_features)
        doc_indice = self._document_sampler.choice(scores).get("choice")
        return doc_indice
    

class ActorModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, user_embed, num_topics, slate_size):
        super().__init__(name="ActorModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._user_embed = user_embed
        self._num_topics = num_topics
        self._slate_size = slate_size
        self._net = tf.keras.Sequential(name="Actor")
        # self._net.add(tf.keras.layers.SimpleRNN(32))  # rnn overfit a lot
        self._net.add(tf.keras.layers.Dense(256,name="actor1"))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(self._num_docs,name="actor2"))

    def call(self, user_vec, available_docs):
        if available_docs==None: 
            scores = self._net(tf.zeros([self._num_users,self._user_embed+self._slate_size*(1+self._num_topics)]))
            return scores
        docs_quality = available_docs.get("doc_quality")
        docs_features = available_docs.get("doc_features")
        input_features = tf.concat([user_vec, tf.reshape(tf.expand_dims(docs_quality,-1),[1,-1]), tf.reshape(docs_features,[1,-1])],-1)
        scores = self._net(input_features)
        return scores


class CriticModel(tf.keras.Model):
    """A tf.keras model that calculate score for each (user, document) pair
    and return the probability of selection in next slate."""

    def __init__(self, num_users, num_docs, user_embed, num_topics, slate_size):
        super().__init__(name="CriticModel")
        self._num_users = num_users
        self._num_docs = num_docs
        self._user_embed = user_embed
        self._num_topics = num_topics
        self._slate_size = slate_size
        self._net = tf.keras.Sequential(name="Critic")
        # self._net.add(tf.keras.layers.SimpleRNN(32))  # rnn overfit a lot
        self._net.add(tf.keras.layers.Dense(256,name="critic1"))
        self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        # self._net.add(tf.keras.layers.Dense(16))
        # self._net.add(tf.keras.layers.LeakyReLU())
        # self._net.add(tf.keras.layers.Dropout(.2))
        self._net.add(
            tf.keras.layers.Dense(1,name="critic2"))

    def call(self, user_vec, chosen_doc_quality, chosen_doc_feature, chosen_doc_length):
        if chosen_doc_quality==None: 
            time = self._net(tf.zeros([self._num_users,self._user_embed+(2+self._num_topics)]))
            return time
        input_features = tf.concat([user_vec, tf.reshape(tf.expand_dims(chosen_doc_length,-1),[1,-1]), tf.reshape(tf.expand_dims(chosen_doc_quality,-1),[1,-1]), tf.reshape(chosen_doc_feature,[1,-1])],-1)
        time = self._net(input_features)
        return time

@gin.configurable
class RepUCBRecommender(recommender.BaseRecommender):
    """A collaborative filtering based recommender implementation. Now using representation of user."""

    def __init__(self,
                 config,
                 phi_model_ctor=RepPhiModel,
                 mu_model_ctor=RepMuModel,
                 actor_model_ctor=ActorModel,
                 critic_model_ctor=CriticModel,
                 name="RepUCBRecommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
        super().__init__(config, name=name)
        self._history_length = config["history_length"]
        self._num_docs = config.get("num_docs")
        self._num_topics = config.get("num_topics")
        self._user_embed = config.get("user_embed")
        self._doc_embed = config.get("doc_embed")
        self._slate_size = config.get("slate_size")
        self._available_documents = None
        
        self._phi_model = phi_model_ctor(self._num_users, self._num_docs, self._num_topics, self._user_embed, self._doc_embed,
                                 self._history_length, self._slate_size)  # doc_dim as hyperparam
        self._mu_model = mu_model_ctor(self._num_users, self._user_embed, self._num_topics, self._slate_size)
        self._actor_model = actor_model_ctor(self._num_users, self._num_docs, self._user_embed, self._num_topics, self._slate_size)
        self._critic_model  = critic_model_ctor(self._num_users, self._num_docs, self._user_embed, self._num_topics, self._slate_size)
        
        slate_doc_quality_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_quality_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_quality_history_model, batch_ndims=1)
        slate_doc_features_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size, self._num_topics,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_features_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_features_history_model, batch_ndims=1)
        choice_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.int32)
        self._choice_history = dynamic.NoOPOrContinueStateModel(
            choice_history_model, batch_ndims=1)
        ctime_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._ctime_history = dynamic.NoOPOrContinueStateModel(
            ctime_history_model, batch_ndims=1)
        self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
            self._slate_size, (self._num_users,),
            -np.Inf * tf.ones(self._num_users))
        # Call model to create weights
        slate_doc_quality_history = self._slate_doc_quality_history.initial_state().get("state")
        slate_doc_features_history = self._slate_doc_features_history.initial_state().get("state")
        ctime_history = self._ctime_history.initial_state().get("state")
        choice_history = self._choice_history.initial_state().get("state")
        user_vec = self._phi_model(slate_doc_quality_history, slate_doc_features_history, choice_history, ctime_history)
        self._mu_model(user_vec, None, None)
        self._actor_model(user_vec, self._available_documents)

    def initial_state(self):
        """The initial state value."""
        slate_doc_quality_history_initial = self._slate_doc_quality_history.initial_state().prefixed_with(
            "slate_doc_quality_history")
        slate_doc_features_history_initial = self._slate_doc_features_history.initial_state().prefixed_with(
            "slate_doc_features_history")
        choice_history_initial = self._choice_history.initial_state().prefixed_with(
            "choice_history")
        ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
            "ctime_history")
        return slate_doc_quality_history_initial.union(slate_doc_features_history_initial).union(choice_history_initial).union(ctime_history_initial)

    def next_state(self, previous_state, user_response,
                   slate_docs):
        # We update histories of only users who chose a doc.
        no_choice = tf.equal(user_response.get("choice"),
                             self._slate_size)[Ellipsis, tf.newaxis]
        next_slate_doc_quality_history = self._slate_doc_quality_history.next_state(
            previous_state.get("slate_doc_quality_history"),
            Value(input=slate_docs.get("doc_quality"),
                  condition=no_choice)).prefixed_with("slate_doc_quality_history")
        next_slate_doc_features_history = self._slate_doc_features_history.next_state(
            previous_state.get("slate_doc_features_history"),
            Value(input=slate_docs.get("doc_features"),
                  condition=no_choice)).prefixed_with("slate_doc_features_history")
        """The state value after the initial value."""
        chosen_doc_idx = user_response.get("choice")
        next_choice_history = self._choice_history.next_state(
            previous_state.get("choice_history"),
            Value(input=chosen_doc_idx,
                  condition=no_choice)).prefixed_with("choice_history")
        # Update consumed time.
        time_consumed = tf.reshape(
            user_response.get("consumed_time"), [self._num_users])
        next_ctime_history = self._ctime_history.next_state(
            previous_state.get("ctime_history"),
            Value(input=time_consumed,
                  condition=no_choice)).prefixed_with("ctime_history")
        return next_slate_doc_quality_history.union(next_slate_doc_features_history).union(next_choice_history).union(next_ctime_history)

    def slate_docs(self, previous_state, user_obs,
                   available_docs):
        """The slate_docs value."""
        del user_obs
        slate_doc_quality_history = previous_state.get("slate_doc_quality_history").get("state")
        slate_doc_features_history = previous_state.get("slate_doc_features_history").get("state")
        ctime_history = previous_state.get("ctime_history").get("state")
        choice_history = previous_state.get("choice_history").get("state")
        user_vec = self._phi_model(slate_doc_quality_history, slate_doc_features_history, choice_history, ctime_history)
        scores = self._actor_model(user_vec, available_docs)
        doc_indices = self._document_sampler.choice(scores).get("choice")
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
        choice_guess = self._mu_model(user_vec, slate.get("doc_quality"), slate.get("doc_features"))
        chosen_doc = selector_lib.get_chosen(slate, choice_guess)
        time_guess = self._critic_model(user_vec, chosen_doc.get("doc_length"), chosen_doc.get("doc_quality"), chosen_doc.get("doc_features"))
        return slate.union(Value(doc_ranks=doc_indices))

    def specs(self):
        state_spec = self._slate_doc_quality_history.specs().prefixed_with("slate_doc_quality_history").union(self._slate_doc_features_history.specs().prefixed_with("slate_doc_features_history")).union(self._choice_history.specs().prefixed_with("choice_history")).union(
            self._ctime_history.specs().prefixed_with("ctime_history"))
        slate_docs_spec = ValueSpec(
            doc_ranks=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones(
                        (self._num_users, self._num_docs)) * self._num_docs)),
            doc_id=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_docs)),
            doc_topic=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_topics)),
            doc_quality=Space(
                spaces.Box(
                    low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
            doc_features=Space(
                spaces.Box(
                    low=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    -np.Inf,
                    high=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    np.Inf)),
            doc_length=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)))
        return state_spec.prefixed_with("state").union(
            slate_docs_spec.prefixed_with("slate"))
    

@gin.configurable
class EstimatedRecommender(recommender.BaseRecommender):
    """A collaborative filtering based recommender implementation. Now using representation of user."""

    def __init__(self,
                 config,
                 phi,
                 mu,
                 actor,
                 critic,
                 name="EstimatedRecommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
        super().__init__(config, name=name)
        self._history_length = config["history_length"]
        self._num_docs = config.get("num_docs")
        self._num_topics = config.get("num_topics")
        self._user_embed = config.get("user_embed")
        self._doc_embed = config.get("doc_embed")
        self._slate_size = config.get("slate_size")
        self._available_documents = None
        
        self._phi_model = phi
        self._mu_model = mu
        self._actor_model = actor
        self._critic_model  = critic
        
        slate_doc_quality_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_quality_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_quality_history_model, batch_ndims=1)
        slate_doc_features_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size, self._num_topics,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_features_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_features_history_model, batch_ndims=1)
        choice_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.int32)
        self._choice_history = dynamic.NoOPOrContinueStateModel(
            choice_history_model, batch_ndims=1)
        ctime_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._ctime_history = dynamic.NoOPOrContinueStateModel(
            ctime_history_model, batch_ndims=1)
        self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
            self._slate_size, (self._num_users,),
            -np.Inf * tf.ones(self._num_users))
        # Call model to create weights
        slate_doc_quality_history = self._slate_doc_quality_history.initial_state().get("state")
        slate_doc_features_history = self._slate_doc_features_history.initial_state().get("state")
        ctime_history = self._ctime_history.initial_state().get("state")
        choice_history = self._choice_history.initial_state().get("state")
        user_vec = self._phi_model(slate_doc_quality_history, slate_doc_features_history, choice_history, ctime_history)
        self._mu_model(user_vec, None, None)
        self._actor_model(user_vec, self._available_documents)

    def initial_state(self):
        """The initial state value."""
        slate_doc_quality_history_initial = self._slate_doc_quality_history.initial_state().prefixed_with(
            "slate_doc_quality_history")
        slate_doc_features_history_initial = self._slate_doc_features_history.initial_state().prefixed_with(
            "slate_doc_features_history")
        choice_history_initial = self._choice_history.initial_state().prefixed_with(
            "choice_history")
        ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
            "ctime_history")
        return slate_doc_quality_history_initial.union(slate_doc_features_history_initial).union(choice_history_initial).union(ctime_history_initial)

    def next_state(self, previous_state, user_response,
                   slate_docs):
        # We update histories of only users who chose a doc.
        no_choice = tf.equal(user_response.get("choice_guess")[0],
                             self._slate_size)[Ellipsis, tf.newaxis]
        next_slate_doc_quality_history = self._slate_doc_quality_history.next_state(
            previous_state.get("slate_doc_quality_history"),
            Value(input=slate_docs.get("doc_quality"),
                  condition=no_choice)).prefixed_with("slate_doc_quality_history")
        next_slate_doc_features_history = self._slate_doc_features_history.next_state(
            previous_state.get("slate_doc_features_history"),
            Value(input=slate_docs.get("doc_features"),
                  condition=no_choice)).prefixed_with("slate_doc_features_history")
        """The state value after the initial value."""
        chosen_doc_idx = user_response.get("choice_guess")[0]
        next_choice_history = self._choice_history.next_state(
            previous_state.get("choice_history"),
            Value(input=chosen_doc_idx,
                  condition=no_choice)).prefixed_with("choice_history")
        # Update consumed time.
        time_consumed = tf.reshape(
            user_response.get("consumed_time"), [self._num_users])
        next_ctime_history = self._ctime_history.next_state(
            previous_state.get("ctime_history"),
            Value(input=time_consumed,
                  condition=no_choice)).prefixed_with("ctime_history")
        return next_slate_doc_quality_history.union(next_slate_doc_features_history).union(next_choice_history).union(next_ctime_history)

    def slate_docs(self, previous_state, user_obs,
                   available_docs):
        """The slate_docs value."""
        del user_obs
        slate_doc_quality_history = previous_state.get("slate_doc_quality_history").get("state")
        slate_doc_features_history = previous_state.get("slate_doc_features_history").get("state")
        ctime_history = previous_state.get("ctime_history").get("state")
        choice_history = previous_state.get("choice_history").get("state")
        user_vec = self._phi_model(slate_doc_quality_history, slate_doc_features_history, choice_history, ctime_history)
        scores = self._actor_model(user_vec, available_docs)
        doc_indices = self._document_sampler.choice(scores).get("choice")
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
        choice_guess = self._mu_model(user_vec, slate.get("doc_quality"), slate.get("doc_features"))
        chosen_doc = selector_lib.get_chosen(slate, choice_guess)
        time_guess = self._critic_model(user_vec, chosen_doc.get("doc_length"), chosen_doc.get("doc_quality"), chosen_doc.get("doc_features"))
        return slate.union(Value(doc_ranks=doc_indices)).union(Value(choice_guess=choice_guess)).union(Value(consumedtime_guess=time_guess)).union(Value(uservec_guess=user_vec))

    def specs(self):
        state_spec = self._slate_doc_quality_history.specs().prefixed_with("slate_doc_quality_history").union(self._slate_doc_features_history.specs().prefixed_with("slate_doc_features_history")).union(self._choice_history.specs().prefixed_with("choice_history")).union(
            self._ctime_history.specs().prefixed_with("ctime_history"))
        slate_docs_spec = ValueSpec(
            uservec_guess=Space(
                spaces.Box(
                    low=np.zeros((self._num_users,self._user_embed)),
                    high=np.ones(
                        (self._num_users,self._user_embed)) * np.Inf)),
            consumedtime_guess=Space(
                spaces.Box(
                    low=np.zeros((self._num_users,)),
                    high=np.ones(
                        (self._num_users,)) * self._slate_size)),
            choice_guess=Space(
                spaces.Box(
                    low=np.zeros((self._num_users,)),
                    high=np.ones(
                        (self._num_users,)) * self._slate_size)),
            doc_ranks=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones(
                        (self._num_users, self._num_docs)) * self._num_docs)),
            doc_id=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_docs)),
            doc_topic=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_topics)),
            doc_quality=Space(
                spaces.Box(
                    low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
            doc_features=Space(
                spaces.Box(
                    low=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    -np.Inf,
                    high=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    np.Inf)),
            doc_length=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)))
        return state_spec.prefixed_with("state").union(
            slate_docs_spec.prefixed_with("slate"))
    

@gin.configurable
class RepresentationRecommender(recommender.BaseRecommender):
    """A collaborative filtering based recommender implementation. Now using representation of user."""

    def __init__(self,
                 config,
                 model_ctor=RepRecModel,
                 name="RepRecRecommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
        super().__init__(config, name=name)
        self._history_length = config["history_length"]
        self._num_docs = config.get("num_docs")
        self._num_topics = config.get("num_topics")
        self._slate_size = config.get("slate_size")
        self._available_documents = None
        self._model = model_ctor(self._num_users, self._num_docs, self._num_topics, self._num_topics,
                                 self._history_length, self._slate_size)  # doc_dim as hyperparam
        slate_doc_quality_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_quality_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_quality_history_model, batch_ndims=1)
        slate_doc_features_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(self._slate_size, self._num_topics,),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._slate_doc_features_history = dynamic.NoOPOrContinueStateModel(
            slate_doc_features_history_model, batch_ndims=1)
        doc_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.int32)
        self._doc_history = dynamic.NoOPOrContinueStateModel(
            doc_history_model, batch_ndims=1)
        ctime_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._ctime_history = dynamic.NoOPOrContinueStateModel(
            ctime_history_model, batch_ndims=1)
        self._document_sampler = selector_lib.IteratedMultinomialLogitChoiceModel(
            self._slate_size, (self._num_users,),
            -np.Inf * tf.ones(self._num_users))
        # Call model to create weights
        slate_doc_quality_history = self._slate_doc_quality_history.initial_state().get("state")
        slate_doc_features_history = self._slate_doc_features_history.initial_state().get("state")
        ctime_history = self._ctime_history.initial_state().get("state")
        docid_history = self._doc_history.initial_state().get("state")
        self._model(slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history, self._available_documents)

    def initial_state(self):
        """The initial state value."""
        slate_doc_quality_history_initial = self._slate_doc_quality_history.initial_state().prefixed_with(
            "slate_doc_quality_history")
        slate_doc_features_history_initial = self._slate_doc_features_history.initial_state().prefixed_with(
            "slate_doc_features_history")
        doc_history_initial = self._doc_history.initial_state().prefixed_with(
            "doc_history")
        ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
            "ctime_history")
        return slate_doc_quality_history_initial.union(slate_doc_features_history_initial).union(doc_history_initial).union(ctime_history_initial)

    def next_state(self, previous_state, user_response,
                   slate_docs):
        # We update histories of only users who chose a doc.
        no_choice = tf.equal(user_response.get("choice"),
                             self._slate_size)[Ellipsis, tf.newaxis]
        next_slate_doc_quality_history = self._slate_doc_quality_history.next_state(
            previous_state.get("slate_doc_quality_history"),
            Value(input=slate_docs.get("doc_quality"),
                  condition=no_choice)).prefixed_with("slate_doc_quality_history")
        next_slate_doc_features_history = self._slate_doc_features_history.next_state(
            previous_state.get("slate_doc_features_history"),
            Value(input=slate_docs.get("doc_features"),
                  condition=no_choice)).prefixed_with("slate_doc_features_history")
        """The state value after the initial value."""
        chosen_doc_idx = user_response.get("choice")
        chosen_doc_features = selector_lib.get_chosen(
            slate_docs, chosen_doc_idx)
        # Update doc_id history.
        doc_consumed = tf.reshape(
            chosen_doc_features.get("doc_id"), [self._num_users])
        next_doc_id_history = self._doc_history.next_state(
            previous_state.get("doc_history"),
            Value(input=doc_consumed,
                  condition=no_choice)).prefixed_with("doc_history")
        # Update consumed time.
        time_consumed = tf.reshape(
            user_response.get("consumed_time"), [self._num_users])
        next_ctime_history = self._ctime_history.next_state(
            previous_state.get("ctime_history"),
            Value(input=time_consumed,
                  condition=no_choice)).prefixed_with("ctime_history")
        return next_slate_doc_quality_history.union(next_slate_doc_features_history).union(next_doc_id_history).union(next_ctime_history)

    def slate_docs(self, previous_state, user_obs,
                   available_docs):
        """The slate_docs value."""
        del user_obs
        slate_doc_quality_history = previous_state.get("slate_doc_quality_history").get("state")
        slate_doc_features_history = previous_state.get("slate_doc_features_history").get("state")
        ctime_history = previous_state.get("ctime_history").get("state")
        docid_history = previous_state.get("doc_history").get("state")
        scores = self._model(slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history, available_docs)
        doc_indices = self._document_sampler.choice(scores).get("choice")
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
        return slate.union(Value(doc_ranks=doc_indices))

    def specs(self):
        state_spec = self._slate_doc_quality_history.specs().prefixed_with("slate_doc_quality_history").union(self._slate_doc_features_history.specs().prefixed_with("slate_doc_features_history")).union(self._doc_history.specs().prefixed_with("doc_history")).union(
            self._ctime_history.specs().prefixed_with("ctime_history"))
        slate_docs_spec = ValueSpec(
            doc_ranks=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones(
                        (self._num_users, self._num_docs)) * self._num_docs)),
            doc_id=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_docs)),
            doc_topic=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_topics)),
            doc_quality=Space(
                spaces.Box(
                    low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
            doc_features=Space(
                spaces.Box(
                    low=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    -np.Inf,
                    high=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    np.Inf)),
            doc_length=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)))
        return state_spec.prefixed_with("state").union(
            slate_docs_spec.prefixed_with("slate"))


@gin.configurable
class UniformRandomRecommender(recommender.BaseRecommender):
    """A collaborative filtering based recommender implementation. Now using uniformly random choices."""

    def __init__(self,
                 config,
                 model_ctor=CollabFilteringModel,
                 name="UniformRandomRecommender"):  # pytype: disable=annotation-type-mismatch  # typed-keras
        super().__init__(config, name=name)
        self._history_length = config["history_length"]
        self._num_docs = config.get("num_docs")
        self._num_topics = config.get("num_topics")
        self._model = model_ctor(self._num_users, self._num_docs, self._num_topics,
                                 self._history_length)
        doc_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.int32)
        self._doc_history = dynamic.NoOPOrContinueStateModel(
            doc_history_model, batch_ndims=1)
        ctime_history_model = estimation.FiniteHistoryStateModel(
            history_length=self._history_length,
            observation_shape=(),
            batch_shape=(self._num_users,),
            dtype=tf.float32)
        self._ctime_history = dynamic.NoOPOrContinueStateModel(
            ctime_history_model, batch_ndims=1)
        self._document_sampler = selector_lib.UniformRandomChoiceModel(
            self._slate_size, self._num_docs, (self._num_users,),
            -np.Inf * tf.ones(self._num_users))
        # Call model to create weights
        ctime_history = self._ctime_history.initial_state().get("state")
        docid_history = self._doc_history.initial_state().get("state")
        self._model(docid_history, ctime_history)

    def initial_state(self):
        """The initial state value."""
        doc_history_initial = self._doc_history.initial_state().prefixed_with(
            "doc_history")
        ctime_history_initial = self._ctime_history.initial_state().prefixed_with(
            "ctime_history")
        return doc_history_initial.union(ctime_history_initial)

    def next_state(self, previous_state, user_response,
                   slate_docs):
        """The state value after the initial value."""
        chosen_doc_idx = user_response.get("choice")
        chosen_doc_features = selector_lib.get_chosen(
            slate_docs, chosen_doc_idx)
        # Update doc_id history.
        doc_consumed = tf.reshape(
            chosen_doc_features.get("doc_id"), [self._num_users])
        # We update histories of only users who chose a doc.
        no_choice = tf.equal(user_response.get("choice"),
                             self._slate_size)[Ellipsis, tf.newaxis]
        next_doc_id_history = self._doc_history.next_state(
            previous_state.get("doc_history"),
            Value(input=doc_consumed,
                  condition=no_choice)).prefixed_with("doc_history")
        # Update consumed time.
        time_consumed = tf.reshape(
            user_response.get("consumed_time"), [self._num_users])
        next_ctime_history = self._ctime_history.next_state(
            previous_state.get("ctime_history"),
            Value(input=time_consumed,
                  condition=no_choice)).prefixed_with("ctime_history")
        return next_doc_id_history.union(next_ctime_history)

    def slate_docs(self, previous_state, user_obs,
                   available_docs):
        """The slate_docs value."""
        del user_obs
        ctime_history = previous_state.get("ctime_history").get("state")
        docid_history = previous_state.get("doc_history").get("state")
        scores = self._model(docid_history, ctime_history)
        doc_indices = self._document_sampler.choice(scores).get("choice")
        slate = available_docs.map(lambda field: tf.gather(field, doc_indices))
        return slate.union(Value(doc_ranks=doc_indices))

    def specs(self):
        state_spec = self._doc_history.specs().prefixed_with("doc_history").union(
            self._ctime_history.specs().prefixed_with("ctime_history"))
        slate_docs_spec = ValueSpec(
            doc_ranks=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._num_docs)),
                    high=np.ones(
                        (self._num_users, self._num_docs)) * self._num_docs)),
            doc_id=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_docs)),
            doc_topic=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones(
                        (self._num_users, self._slate_size)) * self._num_topics)),
            doc_quality=Space(
                spaces.Box(
                    low=np.ones((self._num_users, self._slate_size)) * -np.Inf,
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)),
            doc_features=Space(
                spaces.Box(
                    low=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    -np.Inf,
                    high=np.ones(
                        (self._num_users, self._slate_size, self._num_topics)) *
                    np.Inf)),
            doc_length=Space(
                spaces.Box(
                    low=np.zeros((self._num_users, self._slate_size)),
                    high=np.ones((self._num_users, self._slate_size)) * np.Inf)))
        return state_spec.prefixed_with("state").union(
            slate_docs_spec.prefixed_with("slate"))
