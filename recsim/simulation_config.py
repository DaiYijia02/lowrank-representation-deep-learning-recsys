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

"""Configuration parameters for running recs simulation."""
import functools
from typing import Collection

import gin
import recsim_env.corpus as corpus
from recsim_ng.applications.recsys_partially_observable_rl import metrics
import recsim_env.recommender as recommender
import recsim_env.user as user
# from recsim_ng.applications.recsys_partially_observable_rl import user
from recsim_ng.core import variable
# from recsim_ng.lib.tensorflow import entity # Note: we have modified the entity class as in the file.
import recsim_env.entity as entity # Not using this because of local issues.
import recsim_env.recommendation_simulation as simulation

Variable = variable.Variable


@gin.configurable
def create_representation_simulation_network(
    num_users,
    num_topics,
    num_docs,
    slate_size,
    history_length,
    freeze_corpus=True,
):
    """Returns a network for interests evolution simulation."""
    config = {
        # Common parameters
        'num_users': num_users,
        'num_topics': num_topics,
        'num_docs': num_docs,
        'slate_size': slate_size,
        # History length for user representation in recommender.
        'history_length': history_length,
    }
    if freeze_corpus:
        def corpus_init(config): return corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
            config).initial_state()
        def corpus_ctor(config): return corpus.StaticCorpus(config, corpus_init(config)
                                                            )
    else:
        corpus_ctor = corpus.CorpusWithTopicAndQuality

    def var_fn(): return simulation.recs_story(  # pylint: disable=g-long-lambda
        config, user.InterestEvolutionUser, corpus_ctor,
        functools.partial(recommender.UniformRandomRecommender), metrics.
        ConsumedTimeAsRewardMetrics)
    simulation_vars, trainable_vars = entity.story_with_trainable_variables(
        var_fn)
    return simulation_vars, trainable_vars['Recommender']


@gin.configurable
def create_online_representation_simulation_network(
    num_users,
    num_topics,
    num_docs,
    slate_size,
    history_length,
    freeze_corpus=True,
):
    """Returns a network for interests evolution simulation."""
    config = {
        # Common parameters
        #
        'num_users': num_users,
        'num_topics': num_topics,
        'num_docs': num_docs,
        'slate_size': slate_size,
        # History length for user representation in recommender.
        #
        'history_length': history_length,
    }
    if freeze_corpus:
        def corpus_init(config): return corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
            config).initial_state()
        def corpus_ctor(config): return corpus.StaticCorpus(config, corpus_init(config)
                                                            )
    else:
        corpus_ctor = corpus.CorpusWithTopicAndQuality

    def var_fn(): return simulation.recs_story_for_representation(  # pylint: disable=g-long-lambda
        config, user.InterestEvolutionUser, corpus_ctor,
        functools.partial(recommender.RepresentationRecommender), metrics.
        ConsumedTimeAsRewardMetrics)
    simulation_vars, trainable_vars = entity.story_with_trainable_variables(
        var_fn)
    return simulation_vars, trainable_vars['RepresentationRecommender']


@gin.configurable
def create_rep_ucb_simulation_network(
    num_users,
    num_topics,
    num_docs,
    slate_size,
    history_length,
    freeze_corpus=True,
):
    """Returns a network for interests evolution simulation."""
    config = {
        # Common parameters
        #
        'num_users': num_users,
        'num_topics': num_topics,
        'num_docs': num_docs,
        'slate_size': slate_size,
        # History length for user representation in recommender.
        #
        'history_length': history_length,
        # Model hyperparameters.
        #
        'user_embed': 32,
        'doc_embed': 8,
    }
    if freeze_corpus:
        def corpus_init(config): return corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
            config).initial_state()
        def corpus_ctor(config): return corpus.StaticCorpus(config, corpus_init(config)
                                                            )
    else:
        corpus_ctor = corpus.CorpusWithTopicAndQuality

    def var_fn_1(): return simulation.recs_story_for_representation(  # pylint: disable=g-long-lambda
        config, user.InterestEvolutionUser, corpus_ctor,
        functools.partial(recommender.RepUCBRecommender), metrics.
        ConsumedTimeAsRewardMetrics)
    simulation_vars_rep, trainable_vars, phi, mu, actor, critic = entity.story_with_trainable_variables_and_model(
        var_fn_1)
    
    def var_fn_2(): return simulation.recs_story(  # pylint: disable=g-long-lambda
        config, user.InterestEvolutionUser, corpus_ctor,
        functools.partial(recommender.UniformRandomRecommender), metrics.
        ConsumedTimeAsRewardMetrics)
    simulation_vars_uni, _ = entity.story_with_trainable_variables(
        var_fn_2)
    
    def var_fn_3(): return simulation.estimated_recs_story(  # pylint: disable=g-long-lambda
        config, phi, mu, actor, critic, user.EstimatedUser, corpus_ctor,
        functools.partial(recommender.EstimatedRecommender), metrics.
        ConsumedTimeAsRewardMetrics)
    simulation_vars_est, _, _, _, _, _ = entity.story_with_trainable_variables_and_model(
        var_fn_3)
    
    print("----------------------------")
    print(trainable_vars)
    
    return simulation_vars_rep, simulation_vars_uni, simulation_vars_est, trainable_vars, phi, mu, actor, critic


@gin.configurable
def create_interest_evolution_simulation_network(
    num_users = 1000,
    num_topics = 2,
    num_docs = 100,
    freeze_corpus = True,
    history_length = 15,
):
  """Returns a network for interests evolution simulation."""
  config = {
      # Common parameters
      #
      'num_users': num_users,
      'num_topics': num_topics,
      'num_docs': num_docs,
      'slate_size': 2,
      # History length for user representation in recommender.
      #
      'history_length': history_length,
  }
  if freeze_corpus:
    corpus_init = lambda config: corpus.CorpusWithTopicAndQuality(  # pylint: disable=g-long-lambda
        config).initial_state()
    corpus_ctor = lambda config: corpus.StaticCorpus(config, corpus_init(config)
                                                    )
  else:
    corpus_ctor = corpus.CorpusWithTopicAndQuality
  var_fn = lambda: simulation.recs_story(  # pylint: disable=g-long-lambda
      config, user.InterestEvolutionUser, corpus_ctor,
      functools.partial(recommender.CollabFilteringRecommender), metrics.
      ConsumedTimeAsRewardMetrics)
  simulation_vars, trainable_vars = entity.story_with_trainable_variables(
      var_fn)
  return simulation_vars, trainable_vars['Recommender']