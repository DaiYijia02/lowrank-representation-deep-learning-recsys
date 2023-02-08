"""Agent that implements the Slate-Q algorithms."""
import gin.tf
import numpy as np
from recsim import agent as abstract_agent
from recsim import choice_model
import recsim_dqn_adapt
import tensorflow.compat.v1 as tf


def compute_probs_tf(slate, scores_tf, score_no_click_tf):
    """Computes the selection probability and returns selected index.

    This assumes scores are normalizable, e.g., scores cannot be negative.

    Args:
      slate: a list of integers that represents the video slate.
      scores_tf: a float tensor that stores the scores of all documents.
      score_no_click_tf: a float tensor that represents the score for the action
        of picking no document.

    Returns:
      A float tensor that represents the probabilities of selecting each document
        in the slate.
    """
    all_scores = tf.concat([
        tf.gather(scores_tf, slate),
        tf.reshape(score_no_click_tf, (1, 1))
    ], axis=0)  # pyformat: disable
    all_probs = all_scores / tf.reduce_sum(input_tensor=all_scores)
    return all_probs[:-1]


def score_documents_tf(user_obs,
                       doc_obs,
                       no_click_mass=1.0,
                       is_mnl=False,
                       min_normalizer=-1.0):
    """Computes unnormalized scores given both user and document observations.

    This implements both multinomial proportional model and multinomial logit
      model given some parameters. We also assume scores are based on inner
      products of user_obs and doc_obs.

    Args:
      user_obs: An instance of AbstractUserState.
      doc_obs: A numpy array that represents the observation of all documents in
        the candidate set.
      no_click_mass: a float indicating the mass given to a no click option
      is_mnl: whether to use a multinomial logit model instead of a multinomial
        proportional model.
      min_normalizer: A float (<= 0) used to offset the scores to be positive when
        using multinomial proportional model.

    Returns:
      A float tensor that stores unnormalzied scores of documents and a float
        tensor that represents the score for the action of picking no document.
    """
    user_obs = tf.reshape(user_obs, [1, -1])
    scores = tf.reduce_sum(input_tensor=tf.multiply(user_obs, doc_obs), axis=1)
    all_scores = tf.concat([scores, tf.constant([no_click_mass])], axis=0)
    if is_mnl:
        all_scores = tf.nn.softmax(all_scores)
    else:
        all_scores = all_scores - min_normalizer
    return all_scores[:-1], all_scores[-1]


def score_documents(user_obs,
                    doc_obs,
                    no_click_mass=1.0,
                    is_mnl=False,
                    min_normalizer=-1.0):
    """Computes unnormalized scores given both user and document observations.

    Similar to score_documents_tf but works on NumPy objects.

    Args:
      user_obs: An instance of AbstractUserState.
      doc_obs: A numpy array that represents the observation of all documents in
        the candidate set.
      no_click_mass: a float indicating the mass given to a no click option
      is_mnl: whether to use a multinomial logit model instead of a multinomial
        proportional model.
      min_normalizer: A float (<= 0) used to offset the scores to be positive when
        using multinomial proportional model.

    Returns:
      A float array that stores unnormalzied scores of documents and a float
        number that represents the score for the action of picking no document.
    """
    scores = np.array([])
    for doc in doc_obs:
        scores = np.append(scores, np.dot(user_obs, doc))

    all_scores = np.append(scores, no_click_mass)
    if is_mnl:
        all_scores = choice_model.softmax(all_scores)
    else:
        all_scores = all_scores - min_normalizer
    assert not all_scores[
        all_scores < 0.0], 'Normalized scores have non-positive elements.'
    return all_scores[:-1], all_scores[-1]


def select_slate_optimal(slate_size, s_no_click, s, q):
    """Selects the slate using exhaustive search.

    This algorithm corresponds to the method "OS" in
    Ie et al. https://arxiv.org/abs/1905.12767.

    Args:
      slate_size: int, the size of the recommendation slate.
      s_no_click: float tensor, the score for not clicking any document.
      s: [num_of_documents] tensor, the scores for clicking documents.
      q: [num_of_documents] tensor, the predicted q values for documents.

    Returns:
      [slate_size] tensor, the selected slate.
    """

    num_candidates = s.shape.as_list()[0]

    # Obtain all possible slates given current docs in the candidate set.
    mesh_args = [list(range(num_candidates))] * slate_size
    slates = tf.stack(tf.meshgrid(*mesh_args), axis=-1)
    slates = tf.reshape(slates, shape=(-1, slate_size))

    # Filter slates that include duplicates to ensure each document is picked
    # at most once.
    unique_mask = tf.map_fn(
        lambda x: tf.equal(tf.size(input=x), tf.size(input=tf.unique(x)[0])),
        slates,
        dtype=tf.bool)
    slates = tf.boolean_mask(tensor=slates, mask=unique_mask)

    slate_q_values = tf.gather(s * q, slates)
    slate_scores = tf.gather(s, slates)
    slate_normalizer = tf.reduce_sum(
        input_tensor=slate_scores, axis=1) + s_no_click

    slate_q_values = slate_q_values / tf.expand_dims(slate_normalizer, 1)
    slate_sum_q_values = tf.reduce_sum(input_tensor=slate_q_values, axis=1)
    max_q_slate_index = tf.argmax(input=slate_sum_q_values)
    return


@gin.configurable
class SlateRepUCBAgent(recsim_dqn_adapt.RepUCBAgentRecSim,
                       abstract_agent.AbstractEpisodicRecommenderAgent):
    """A recommender agent implements Rep-UCB using DQN and slate decomposition techniques."""

    def __init__(self,
                 sess,
                 observation_space,
                 action_space,
                 optimizer_name='',
                 select_slate_fn=None,
                 compute_target_fn=None,
                 stack_size=1,
                 eval_mode=False,
                 **kwargs):
        """Initializes SlateRepUCBAgent.

        Args:
          sess: a Tensorflow session.
          observation_space: A gym.spaces object that specifies the format of
            observations.
          action_space: A gym.spaces object that specifies the format of actions.
          optimizer_name: The name of the optimizer.
          select_slate_fn: A function that selects the slate.
          compute_target_fn: A function that omputes the target q value. TODO
          stack_size: The stack size for the replay buffer.
          eval_mode: A bool for whether the agent is in training or evaluation mode.
          **kwargs: Keyword arguments to the DQNAgent.
        """
        self._response_adapter = recsim_dqn_adapt.ResponseAdapter(
            observation_space.spaces['response'])
        response_names = self._response_adapter.response_names
        expected_response_names = ['click', 'watch_time']
        if not all(key in response_names for key in expected_response_names):
            raise ValueError(
                "Couldn't find all fields needed for the decomposition: %r" %
                expected_response_names)

        self._click_response_index = response_names.index('click')
        self._reward_response_index = response_names.index('watch_time')
        self._quality_response_index = response_names.index('quality')
        self._cluster_id_response_index = response_names.index('cluster_id')

        self._env_action_space = action_space
        self._num_candidates = int(action_space.nvec[0])
        abstract_agent.AbstractEpisodicRecommenderAgent.__init__(
            self, action_space)

        # The doc score is a [num_candidates] vector.
        self._doc_affinity_scores_ph = tf.placeholder(
            tf.float32, (self._num_candidates,), name='doc_affinity_scores_ph')
        self._prob_no_click_ph = tf.placeholder(
            tf.float32, (), name='prob_no_click_ph')

        self._select_slate_fn = select_slate_fn
        self._compute_target_fn = compute_target_fn  # TODO

        recsim_dqn_adapt.RepUCBAgentRecSim.__init__(
            self,
            sess,
            observation_space,
            num_actions=0,  # Unused.
            stack_size=1,
            optimizer_name=optimizer_name,
            eval_mode=eval_mode,
            **kwargs)

    # The following functions defines how the agent takes actions.

    def step(self, reward, observation):
        """Records the transition and returns the agent's next action.

        It uses document-level user response instead of overral reward as the reward
        of the problem.

        Args:
          reward: unused.
          observation: a space.Dict that includes observation of the user state
            observation, documents and user responses.

        Returns:
          Array, the selected action.
        """
        del reward  # Unused argument.

        responses = observation['response']
        self._raw_observation = observation
        return super(SlateRepUCBAgent,
                     self).step(self._response_adapter.encode(responses),
                                self._obs_adapter.encode(observation))

    def _build_select_slate_op(self):
        p_no_click = self._prob_no_click_ph
        p = self._doc_affinity_scores_ph
        q = self._net_outputs.q_values[0]
        with tf.name_scope('select_slate'):
            self._output_slate = self._select_slate_fn(self._slate_size, p_no_click,
                                                       p, q)

        self._output_slate = tf.Print(
            self._output_slate, [tf.constant(
                'cp 1'), self._output_slate, p, q],
            summarize=10000)
        self._output_slate = tf.reshape(
            self._output_slate, (self._slate_size,))

        self._action_counts = tf.get_variable(
            'action_counts',
            shape=[self._num_candidates],
            initializer=tf.zeros_initializer())
        output_slate = tf.reshape(self._output_slate, [-1])
        output_one_hot = tf.one_hot(output_slate, self._num_candidates)
        update_ops = []
        for i in range(self._slate_size):
            update_ops.append(tf.assign_add(
                self._action_counts, output_one_hot[i]))
        self._select_action_update_op = tf.group(*update_ops)

    def _select_action(self):
        """Selects an slate based on the trained model.
        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates. It will
        pick the top slate_size documents with highest Q values and return them as a
        slate.
        Returns:
          Array, the selected action.
        """
        if self.eval_mode:
            epsilon = self.epsilon_eval
        else:
            epsilon = self.epsilon_fn(self.epsilon_decay_period, self.training_steps,
                                      self.min_replay_history, self.epsilon_train)
            self._add_summary('epsilon', epsilon)

        if np.random.random() <= epsilon:
            # Sample without replacement.
            return np.random.choice(
                self._num_candidates, self._slate_size, replace=False)
        else:
            observation = self._raw_observation
            user_obs = observation['user']
            doc_obs = np.array(list(observation['doc'].values()))
            tf.logging.debug('cp 1: %s, %s', doc_obs, observation)
            # TODO(cwhsu): Use score_documents_tf() and remove score_documents().
            scores, score_no_click = score_documents(user_obs, doc_obs)
            output_slate, _ = self._sess.run(
                [self._output_slate, self._select_action_update_op], {
                    self.state_ph: self.state,
                    self._doc_affinity_scores_ph: scores,
                    self._prob_no_click_ph: score_no_click,
                })

            return output_slate
