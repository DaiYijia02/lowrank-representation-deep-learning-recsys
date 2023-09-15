"""User entity for long-term interests evolution simulation."""
from gymnasium import spaces
import torch
import numpy as np
from scipy.special import softmax
from gym_env.utils import sigmoid

class InterestEvolutionUser():
    """Dynamics of a user whose interests evolve over time."""

    def __init__(
        self, user_dim,
        # Step size for updating user interests based on consumed documents
        # (small!). We may want to have different values for different interests
        # to represent how malleable those interests are, e.g., strong dislikes
        # may be less malleable).
        interest_step_size=0.01, max_user_affinity=5.0,
        ):
        self._user_dim = user_dim
        self._interest_step_size = interest_step_size
        self._max_user_affinity = max_user_affinity
        self._init_state = np.random.randn(user_dim)*max_user_affinity
        self._current_state = self._init_state

    def reset(self):
        """Reset a initial state value."""
        self._init_state = np.random.randn(self._user_dim)*self._max_user_affinity
        self._current_state = self._init_state

    def next_state(self, chosen_item):
        """The state value after one-step transition."""
        assert len(chosen_item)==self._user_dim+1
        item_quality = chosen_item[-1]
        item_feature = chosen_item[:-1]

        # User interests are increased/decreased towards the consumed document's
        # topic proportinal to the document quality.
        direction = item_quality * (item_feature - self._current_state)

        # We squash the interest vector to avoid infinite blow-up using the function
        # 4 * M * (sigmoid(X/M) - 0.5) which is roughly linear around the origin and
        # softly saturates at +/-2M. These constants are not intended to be tunable.
        next_state = 4.0 * self._max_user_affinity *(sigmoid(direction*self._interest_step_size/self._max_user_affinity) -
             0.5)
        self._current_state = next_state

    def respond(self, slate_docs):
        """The response from current state."""
        assert len(slate_docs[0])==self._user_dim+1
        affinities = np.apply_along_axis(lambda x: np.inner(x[:-1],self._current_state), 0, slate_docs)
        probabilities = softmax(affinities)
        choice = np.random.choice([i for i in range(len(slate_docs))], p=probabilities)
        
        # Calculate consumption time. High quality documents generate more
        # engagement and lead to positive interest evolution.
        doc_quality = slate_docs[choice][-1]
        satisfaction = doc_quality*affinities[choice]
        return choice, satisfaction
    
    def obs(self):
        """The current state value."""
        return self._current_state

