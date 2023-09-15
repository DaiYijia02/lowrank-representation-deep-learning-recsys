"""Corpus entity for partially observable RL simulation."""

from gymnasium import spaces
import numpy as np

class CorpusWithTopicAndQuality():
  """Defines a corpus with static topic and quality distributions."""

  def __init__(self,
               item_dim,
               num_items,
               topic_max_utility = 2.):
    self._num_topics = item_dim
    self._corpus_size = num_items
    self._topic_max_utility = topic_max_utility
    self._corpus = np.zeros((num_items, item_dim+1))

  def sample(self):
    self._corpus = np.random.randn(self._corpus_size, self._num_topics+1)



