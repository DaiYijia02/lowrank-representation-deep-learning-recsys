import time
from absl import app
import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
import tensorflow_probability as tfp
import online_representation_simulation
import wandb

tfd = tfp.distributions
Value = value.Value

# wandb.init(
#     # set the wandb project where this run will be logged
#     project="recsys_simulation",
    
#     # track hyperparameters and run metadata
#     config={
#     "learning_rate": 3e-4,
#     # "architecture": "CNN",
#     # "dataset": "CIFAR-100",
#     "epochs": 10,
#     }
# )

def main(argv):
    del argv
    num_users = 1
    num_topics = 2
    num_docs = 2
    slate_size = 2
    history_length = 10
    horizon = 50

    # tf.debugging.set_log_device_placement(True)

    simulation_variables_rep, simlulation_variable_uni, simulation_variable_est, trainable_variables, phi, mu, actor, critic = simulation_config.create_rep_ucb_simulation_network(
        num_users=num_users, num_topics=num_topics, num_docs=num_docs, slate_size=slate_size, history_length=history_length)
    online_representation_simulation.run_simulation(
      num_training_steps=10,
      horizon=horizon,
      history_length=history_length,
      global_batch=num_users,
      learning_rate=3e-4,
      simulation_variables_main=simulation_variables_rep,
      simulation_variables_sub=simlulation_variable_uni,
      simulation_variables_est=simulation_variable_est,
      trainable_variables=trainable_variables,
      metric_to_optimize='cumulative_reward',
      phi=phi,
      mu=mu,
      actor=actor,
      critic=critic)

if __name__ == '__main__':
    app.run(main)
