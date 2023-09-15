from absl import app
from recsim_ng.applications.recsys_partially_observable_rl import interest_evolution_simulation
from recsim_ng.applications.recsys_partially_observable_rl import simulation_config


def main(argv):
  del argv
  num_users = 1000
  variables, trainable_variables = (
      simulation_config.create_interest_evolution_simulation_network(
          num_users=num_users))

  interest_evolution_simulation.run_simulation(
      num_training_steps=100,
      horizon=100,
      global_batch=num_users,
      learning_rate=1e-4,
      simulation_variables=variables,
      trainable_variables=trainable_variables,
      metric_to_optimize='cumulative_reward')


if __name__ == '__main__':
  app.run(main)