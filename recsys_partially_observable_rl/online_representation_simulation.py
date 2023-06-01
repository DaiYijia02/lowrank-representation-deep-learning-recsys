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

# Line as: python3
"""WIP: For testing differentiable interest evolution networks."""

from typing import Any, Callable, Collection, Sequence, Text, Optional

from recsim_ng.core import network as network_lib
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import log_probability
import recsim_env.runtime as runtime
from recsim_env.recommender import RepUCBModel
import tensorflow as tf

Network = network_lib.Network
Variable = variable.Variable


def reset_optimizer(learning_rate):
  return tf.keras.optimizers.SGD(learning_rate)

def fn_first_traj(main_traj, sub_traj):
  main_slate_doc_quality_history = main_traj['slate docs'].get('doc_quality')
  main_slate_doc_features_history = main_traj['slate docs'].get('doc_features')
  main_docid_history = main_traj['slate docs'].get('doc_id')
  main_ctime_history = main_traj['user response'].get('consumed_time')
  sub_slate_doc_quality_history = sub_traj['slate docs'].get('doc_quality')[0:1,:,:]
  sub_slate_doc_features_history = sub_traj['slate docs'].get('doc_features')[0:1,:,:,:]
  sub_docid_history = sub_traj['slate docs'].get('doc_id')[0:1,:,:]
  sub_ctime_history = sub_traj['user response'].get('consumed_time')[0:1,:]
  slate_doc_quality_history = tf.concat([main_slate_doc_quality_history, sub_slate_doc_quality_history],0)
  slate_doc_features_history = tf.concat([main_slate_doc_features_history, sub_slate_doc_features_history],0)
  docid_history = tf.concat([main_docid_history, sub_docid_history],0)
  ctime_history = tf.concat([main_ctime_history, sub_ctime_history],0)
  return slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history


def fn_second_traj(main_traj, sub_traj):
  main_slate_doc_quality_history = main_traj['slate docs'].get('doc_quality')[1:,:,:]
  main_slate_doc_features_history = main_traj['slate docs'].get('doc_features')[1:,:,:,:]
  main_docid_history = main_traj['slate docs'].get('doc_id')[1:,:,:]
  main_ctime_history = main_traj['user response'].get('consumed_time')[1:,:]
  sub_slate_doc_quality_history = sub_traj['slate docs'].get('doc_quality')
  sub_slate_doc_features_history = sub_traj['slate docs'].get('doc_features')
  sub_docid_history = sub_traj['slate docs'].get('doc_id')
  sub_ctime_history = sub_traj['user response'].get('consumed_time')
  slate_doc_quality_history = tf.concat([main_slate_doc_quality_history, sub_slate_doc_quality_history],0)
  slate_doc_features_history = tf.concat([main_slate_doc_features_history, sub_slate_doc_features_history],0)
  docid_history = tf.concat([main_docid_history, sub_docid_history],0)
  ctime_history = tf.concat([main_ctime_history, sub_ctime_history],0)
  return slate_doc_quality_history, slate_doc_features_history, docid_history, ctime_history

def fn_last_state(traj):
  def fn(field_value):
    field_value = field_value[-1,...]
    return field_value
  ans = dict()
  for k,v in traj.items():
    ans[k] = v.map(fn)
  return ans

def distributed_train_step(
    tf_runtime_main,
    tf_runtime_sub,
    horizon,
    global_batch,
    trainable_variables,
    trajs_list_1,
    trajs_list_2,
    metric_to_optimize='reward',
    optimizer = None
    
):
  """Extracts gradient update and training variables for updating network."""
  with tf.GradientTape() as tape:
    # Rep-UCB leaves two last-step to uniform recommender
    main_traj = tf_runtime_main.trajectory(length=horizon - 1)
    # tf.print(main_traj['slate docs'].__str__())
    # tf.print(main_traj['user response'].__str__())
    # tf.print(main_traj['user state'].__str__())
    last_state = fn_last_state(main_traj)
    sub_traj = tf_runtime_sub.sub_trajectory(length=2, starting_value=last_state)
    first_traj = fn_first_traj(main_traj, sub_traj)
    second_traj = fn_second_traj(main_traj, sub_traj)
    trajs_list_1.append(first_traj)
    trajs_list_2.append(second_traj)
    
    # Train argmax Phi and Mu with the new trajs

    last_state = tf_runtime_main.execute(num_steps=horizon - 1)
    last_metric_value = last_state['metrics state'].get(metric_to_optimize)
    log_prob = last_state['slate docs_log_prob_accum'].get('doc_ranks')
    # print(log_prob)
    objective = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob, 1)
    objective /= float(global_batch)

  print(trainable_variables)
  grads = tape.gradient(objective, trainable_variables)
  if optimizer:
    grads_and_vars = list(zip(grads, trainable_variables))
    optimizer.apply_gradients(grads_and_vars)
  return grads, objective, tf.reduce_mean(last_metric_value)


def make_runtime(variables):
  """Makes simulation + policy log-prob runtime."""
  variables = list(variables)
  slate_var = [var for var in variables if 'slate docs' == var.name]
  log_prob_var = log_probability.log_prob_variables_from_direct_output(
      slate_var)
  accumulator = log_probability.log_prob_accumulator_variables(log_prob_var)
  tf_runtime = runtime.TFRuntime(
      network=network_lib.Network(
          variables=list(variables) + list(log_prob_var) + list(accumulator)),
      graph_compile=False)
  return tf_runtime


def make_train_step(
    tf_runtime_main,
    tf_runtime_sub,
    horizon,
    global_batch,
    trainable_variables,
    metric_to_optimize,
    trajs_list_1,
    trajs_list_2,
    optimizer = None
):
  """Wraps a traced training step function for use in learning loops."""

  @tf.function
  def distributed_grad_and_train():
    return distributed_train_step(tf_runtime_main, tf_runtime_sub, horizon, global_batch,
                                  trainable_variables, trajs_list_1, trajs_list_2, metric_to_optimize,
                                  optimizer)

  return distributed_grad_and_train


def run_simulation(
    num_training_steps,
    horizon,
    global_batch,
    learning_rate,
    simulation_variables_main,
    simulation_variables_sub,
    trainable_variables,
    metric_to_optimize = 'reward',
):
  """Runs simulation over multiple horizon steps while learning policy vars."""
  trajs_list_1 = []
  trajs_list_2 = []

  optimizer = reset_optimizer(learning_rate)
  tf_runtime_rep = make_runtime(simulation_variables_main)
  tf_runtime_uni = make_runtime(simulation_variables_sub)
  train_step = make_train_step(tf_runtime_rep, tf_runtime_uni, horizon, global_batch,
                               trainable_variables, metric_to_optimize, trajs_list_1, trajs_list_2,
                               optimizer)

  metric_list = []
  for _ in range(num_training_steps):
    _, _, last_metric = train_step()
    metric_list.append(last_metric)

