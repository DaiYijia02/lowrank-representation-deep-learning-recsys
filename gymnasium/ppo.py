import gymnasium as gym
from stable_baselines3 import PPO
from gym_env.user import InterestEvolutionUser
from gym_env.corpus import CorpusWithTopicAndQuality
import wandb

wandb.init(
    project="dynamic-recsys-ppo",

    config={
    "learning_rate": None,
    "agent": "PPO",
    "epochs": 10e13,
    }
)

env = dict(
    user_dim=2, 
    num_item=3, 
    item_dim=2, 
    hist_seq_len=5, 
    slate_size=2,
    user_state_model=InterestEvolutionUser, 
    corpus_model=CorpusWithTopicAndQuality,
)

env = gym.make('dynamic_recsys-v0', max_episode_steps=10e13, **env)

model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10e13)


obs, info = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, truncated, info = env.step(action)
    wandb.log({"rewards": rewards})

wandb.finish()