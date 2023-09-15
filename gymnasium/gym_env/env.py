from collections import deque
import numpy as np

import gymnasium as gym
from gymnasium import spaces


class DynamicUserRecsysEnv(gym.Env):
    metadata = {
        'name': 'dynamic_user_recsys',
        'render_modes': ['human', 'rgb_array'],
    }

    """User dynamically influenced by the item clicked.
    
    Args:
        user_dim: int, the dimension of user representation;
        num_item: int, the number of items;
        item_dim: int, the dimension of item representation;
        hist_seq_len: int, the length of history cached;
        slate_size: int, the slate size;
        user_state_model: model, the user state and choice;
        corpus_model: model, the item set;
        reward_model: model, the reward model;
        """
    def __init__(
        self, user_dim, num_item, item_dim, 
        hist_seq_len, slate_size,
        user_state_model, corpus_model,
    ):
        self._user_dim = user_dim
        self._num_item = num_item
        self._item_dim = item_dim
        self._hist_seq_len = hist_seq_len
        self._slate_size = slate_size
        self._user_state_model = user_state_model(user_dim)
        self._corpus_model = corpus_model(item_dim, num_item)
        self.item_set = self._corpus_model.sample()

        self.nan_item = np.zeros(item_dim+1)
        self.hist_seq = deque([[self.nan_item]*(slate_size+2)]*hist_seq_len, maxlen=hist_seq_len)  # FIFO que for user's historical interactions
        assert len(self.hist_seq) == hist_seq_len

        # NOTE: each element in 'self.hist_seq' represents [recommended slate items, chosen item]
        # TODO: if we also want to observe the satisfaction score (linear to affinity)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(hist_seq_len,slate_size+2,item_dim+1), 
            dtype=np.float64
        )

        # NOTE: each item has its feature dimension 'item_dim' with an additional quality scalar
        self.action_space = spaces.Box(
            low=-10., 
            high=10., 
            shape=(slate_size,item_dim+1), 
            dtype=np.float64
        )

        # ----------visualization----------
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0
        self.viewer = None
        self.fig, self.axes = None, None
        self.rng = np.random.default_rng()
    
    def reset(self, seed=None):
        self.hist_seq = deque([[self.nan_item]*(self._slate_size+2)]*self._hist_seq_len, maxlen=self._hist_seq_len)
        self._user_state_model.reset()
        self.user_logs = []
        self.rs_logs = []
        self.timestep = 0
        info = {}
        return np.asarray(self.hist_seq), info

    def step(self, action):
        assert action in self.action_space
        # assert np.unique(action).size == len(action), 'repeated items in slate are not allowed!'
        # append a skip-item at the end of the slate to allow user to skip the slate
        # pre-trained reward model will give a learned reward for skipping
        action = [*action, self.nan_item]

        # user chooses from the slate, and give a feedback
        choice, satisfaction = self._user_state_model.respond(action)

        # update user state
        self._user_state_model.next_state(action[choice])
        hist = [*action, action[choice]]
        self.hist_seq.append(hist)

        # ----------visualization----------
        self.timestep += 1
        self.user_logs.append({
            'timestep': self.timestep,
            'user_choice': action[choice], # NOTE: include skip pseudo-item
            'user_next_state': self._user_state_model.obs(),
            'satisfaction': satisfaction
        })
        self.rs_logs.append({
            'timestep': self.timestep,
            'slate': action # NOTE: include skip pseudo-item
        })

        # for gym framework
        obs = np.array(self.hist_seq)
        reward = satisfaction
        done = False
        truncated = False
        info = {
            'timestep': self.timestep,
            'slate': action,
            'user_choice': action[choice],
            'satisfaction': satisfaction,
            'user_next_state': self._user_state_model.obs()
        }
        return obs, reward, done, truncated, info
    
    # ----------VISUALIZATION----------
    def _get_img(self):
        pass

    def render(self, mode='human'):
        pass