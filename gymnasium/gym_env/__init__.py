import gymnasium as gym
# import env
# from user import InterestEvolutionUser
# from corpus import CorpusWithTopicAndQuality

gym.register(
     id='dynamic_recsys-v0',
     entry_point='gym_env.env:DynamicUserRecsysEnv'
)

# env = dict(
#     user_dim=2, 
#     num_item=3, 
#     item_dim=2, 
#     hist_seq_len=5, 
#     slate_size=2,
#     user_state_model=InterestEvolutionUser, 
#     corpus_model=CorpusWithTopicAndQuality,
# )

# env = gym.make('dynamic_recsys-v0', max_episode_steps=1000, **env)

# observation = env.reset()

# for _ in range(1000):
#     action = env.action_space.sample()  # agent policy that uses the observation and info
#     observation, reward, terminated, truncated, info = env.step(action)
#     if terminated:
#         observation, info = env.reset()

# env.close()