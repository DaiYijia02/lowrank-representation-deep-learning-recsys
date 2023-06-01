import time
from absl import app
import simulation_config
from recsim_ng.core import network as network_lib
from recsim_ng.core import value
from recsim_ng.lib.tensorflow import log_probability
from recsim_ng.lib.tensorflow import runtime
import tensorflow as tf
import tensorflow_probability as tfp
import pickle

tfd = tfp.distributions
Value = value.Value


def main(argv):
    del argv
    num_users = 1
    num_topics = 2
    num_docs = 20
    slate_size = 2
    history_length = 20
    num_iters = 500000

    variables, trainable_variables = simulation_config.create_representation_simulation_network(
        num_users=num_users, num_topics=num_topics, num_docs=num_docs, slate_size=slate_size, history_length=history_length)
    data_generation_network = network_lib.Network(variables=variables)
    tf_runtime = runtime.TFRuntime(network=data_generation_network)

    def sample_traj(num_iters):
        for i in range(num_iters):
            traj_file = open(
                'recsim_dataset/data_0.30/traj_'+str(i)+'.pkl', 'wb')
            traj = dict(tf_runtime.trajectory(length=history_length))
            pickle.dump(traj, traj_file)
            traj_file.close()

    sample_traj(num_iters)


if __name__ == '__main__':
    app.run(main)
