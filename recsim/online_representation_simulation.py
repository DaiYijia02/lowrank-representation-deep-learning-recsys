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
import random

from recsim_ng.core import network as network_lib
from recsim_ng.core import variable
from recsim_ng.lib.tensorflow import log_probability
import recsim_env.runtime as runtime
import tensorflow as tf
import wandb

Network = network_lib.Network
Variable = variable.Variable


def reset_optimizer(learning_rate):
  return tf.keras.optimizers.SGD(learning_rate)

def fn_first_traj(main_traj, sub_traj):
  main_slate_doc_quality_history = main_traj['slate docs'].get('doc_quality')
  main_slate_doc_features_history = main_traj['slate docs'].get('doc_features')
  # main_docid_history = main_traj['slate docs'].get('doc_id')
  main_ctime_history = main_traj['user response'].get('consumed_time')
  main_choice_history = main_traj['user response'].get('choice')
  
  sub_slate_doc_quality_history = sub_traj['slate docs'].get('doc_quality')[0:1,:,:]
  sub_slate_doc_features_history = sub_traj['slate docs'].get('doc_features')[0:1,:,:,:]
  # sub_docid_history = sub_traj['slate docs'].get('doc_id')[0:1,:,:]
  sub_ctime_history = sub_traj['user response'].get('consumed_time')[0:1,:]
  sub_choice_history = sub_traj['user response'].get('choice')[0:1,:]
  
  slate_doc_quality_history = tf.concat([main_slate_doc_quality_history, sub_slate_doc_quality_history],0)
  slate_doc_quality_history = tf.expand_dims(tf.squeeze(slate_doc_quality_history),axis=0)
  
  slate_doc_features_history = tf.concat([main_slate_doc_features_history, sub_slate_doc_features_history],0)
  slate_doc_features_history = tf.expand_dims(tf.squeeze(slate_doc_features_history),axis=0)

  # docid_history = tf.concat([main_docid_history, sub_docid_history],0)
  # docid_history = tf.expand_dims(tf.squeeze(docid_history),axis=0)
  
  ctime_history = tf.concat([main_ctime_history, sub_ctime_history],0)
  ctime_history = tf.expand_dims(tf.squeeze(ctime_history),axis=0)
  
  choice_history = tf.concat([main_choice_history, sub_choice_history],0)
  choice_history = tf.expand_dims(tf.squeeze(choice_history),axis=0)

  return slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history


def fn_second_traj(main_traj, sub_traj):
  main_slate_doc_quality_history = main_traj['slate docs'].get('doc_quality')[1:,:,:]
  main_slate_doc_features_history = main_traj['slate docs'].get('doc_features')[1:,:,:,:]
  # main_docid_history = main_traj['slate docs'].get('doc_id')[1:,:,:]
  main_ctime_history = main_traj['user response'].get('consumed_time')[1:,:]
  main_choice_history = main_traj['user response'].get('choice')[1:,:]
  sub_slate_doc_quality_history = sub_traj['slate docs'].get('doc_quality')
  sub_slate_doc_features_history = sub_traj['slate docs'].get('doc_features')
  # sub_docid_history = sub_traj['slate docs'].get('doc_id')
  sub_ctime_history = sub_traj['user response'].get('consumed_time')
  sub_choice_history = sub_traj['user response'].get('choice')

  slate_doc_quality_history = tf.concat([main_slate_doc_quality_history, sub_slate_doc_quality_history],0)
  slate_doc_quality_history = tf.expand_dims(tf.squeeze(slate_doc_quality_history),axis=0)
  
  slate_doc_features_history = tf.concat([main_slate_doc_features_history, sub_slate_doc_features_history],0)
  slate_doc_features_history = tf.expand_dims(tf.squeeze(slate_doc_features_history),axis=0)

  # docid_history = tf.concat([main_docid_history, sub_docid_history],0)
  # docid_history = tf.expand_dims(tf.squeeze(docid_history),axis=0)
  
  ctime_history = tf.concat([main_ctime_history, sub_ctime_history],0)
  ctime_history = tf.expand_dims(tf.squeeze(ctime_history),axis=0)
  
  choice_history = tf.concat([main_choice_history, sub_choice_history],0)
  choice_history = tf.expand_dims(tf.squeeze(choice_history),axis=0)

  return slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history

def fn_last_state(traj):
  def fn(field_value):
    field_value = field_value[-1,...]
    return field_value
  ans = dict()
  for k,v in traj.items():
    ans[k] = v.map(fn)
  return ans

def update_phi(trajs_list_1, trajs_list_2, available_docs, phi, mu, actor, critic, optimizer):
  loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
  trainable_vars = phi.trainable_variables+mu.trainable_variables

  def loss(phi, mu, traj, training):
    # training=training is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history = traj
    y = tf.nn.softmax(tf.one_hot(choice_history[0, -1], 3)) # get the last choice and project it to slate
    user_vec = phi(slate_doc_quality_history[:, :-1], slate_doc_features_history[:, :-1], choice_history[:, :-1], ctime_history[:, :-1], training=training)
    y_ = tf.nn.softmax(tf.one_hot(tf.squeeze(mu(user_vec, slate_doc_quality_history[:, -1:], slate_doc_features_history[:, -1:], training=training)), 3))
    return loss_object(y_true=y, y_pred=y_)

  def grad(phi, mu, traj):
    with tf.GradientTape() as tape2:
      tape2.watch(traj)
      loss_value = loss(phi, mu, traj, training=True)
      
    return loss_value, tape2.gradient(loss_value, trainable_vars)
  
  num_epochs = 10

  for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()

    for traj in trajs_list_1:
      loss_value, grads = grad(phi, mu, traj)
      epoch_loss_avg.update_state(loss_value)
      optimizer.apply_gradients(zip(grads, trainable_vars))
      
loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

def loss(phi, mu, traj, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history = traj
  y = tf.nn.softmax(tf.one_hot(choice_history[0, -1], 3)) # get the last choice and project it to slate
  user_vec = phi(slate_doc_quality_history[:, :-1], slate_doc_features_history[:, :-1], choice_history[:, :-1], ctime_history[:, :-1], training=training)
  y_ = tf.nn.softmax(tf.one_hot(tf.squeeze(mu(user_vec, slate_doc_quality_history[:, -1:], slate_doc_features_history[:, -1:], training=training)), 3))
  return loss_object(y_true=y, y_pred=y_)

def grad(phi, mu, traj, trainable_vars):
  with tf.GradientTape() as tape2:
    tape2.watch(traj)
    loss_value = loss(phi, mu, traj, training=True)
    # print(tape2.gradient(loss_value, trainable_vars))
  return loss_value, tape2.gradient(loss_value, trainable_vars)
  

def distributed_train_step(
    tf_runtime_main,
    tf_runtime_sub,
    tf_runtime_est,
    horizon,
    history_length,
    global_batch,
    trainable_variables,
    trajs_list_1,
    trajs_list_2,
    phi,
    mu,
    actor,
    critic,
    metric_to_optimize='cumulative_reward',
    optimizer = None
    
):
  """Extracts gradient update and training variables for updating network."""
  # loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  # print("Loss test: {}".format(tf.get_static_value(loss_object([[1,0]],[[.5,0.5]]))))
  last_state=None
  # with tf.GradientTape(persistent=True) as tape:
  with tf.GradientTape() as tape:
    # Rep-UCB leaves two last-step to uniform recommender
    # seed = random.randint(horizon, history_length-2)
    main_traj = tf_runtime_main.trajectory(length=history_length)
    # tf.print(main_traj['slate docs'].__str__())
    # tf.print(main_traj['user response'].__str__())
    # tf.print(main_traj['user state'].__str__()
    last_state = fn_last_state(main_traj)
    sub_traj = tf_runtime_sub.sub_trajectory(length=2, starting_value=last_state)
    # first_traj = fn_first_traj(main_traj, sub_traj)
    # second_traj = fn_second_traj(main_traj, sub_traj)
    # trajs_list_1.append(first_traj)
    # trajs_list_2.append(second_traj)

    # available_docs = main_traj['available docs']

    # try out
    # slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history = first_traj
    # y = tf.nn.softmax(tf.one_hot(choice_history[0, -1], 3)) # get the last choice and project it to slate
    # user_vec = phi(slate_doc_quality_history[:, :-1], slate_doc_features_history[:, :-1], choice_history[:, :-1], ctime_history[:, :-1], training=True)
    # y_ = tf.nn.softmax(tf.one_hot(tf.squeeze(mu(user_vec, slate_doc_quality_history[:, -1:], slate_doc_features_history[:, -1:], training=True)), 3))

    # slate_doc_quality_history, slate_doc_features_history, ctime_history, choice_history = first_traj

    # y = tf.nn.softmax(tf.one_hot(tf.stop_gradient(first_traj[3][0, -1]), 3)) # get the last choice and project it to slate
    # user_vec = phi(tf.stop_gradient(first_traj[0][:, :-1]), tf.stop_gradient(first_traj[1][:, :-1]), tf.stop_gradient(first_traj[3][:, :-1]), 
    #                tf.stop_gradient(first_traj[2][:, :-1]), training=True)
    # y_ = tf.nn.softmax(tf.one_hot(tf.squeeze(mu(user_vec, tf.stop_gradient(first_traj[0][:, -1:]), tf.stop_gradient(first_traj[1][:, -1:]), training=True)), 3))
    # objective = loss_object(y_true=y, y_pred=y_)
    # print("Loss test: {}".format(tf.get_static_value(objective)))
    # trainable_variables = phi.trainable_variables+mu.trainable_variables
    # trainable_variables = phi.trainable_variables+mu.trainable_variables
    # # Train argmax Phi and Mu with the new trajs
    # update_phi(trajs_list_1, trajs_list_2, available_docs, phi, mu, actor, critic, optimizer)


    # last_state = tf_runtime_main.execute(num_steps=horizon - 1)
    last_state = fn_last_state(sub_traj)
    last_metric_value = last_state['metrics state'].get(metric_to_optimize)
    # log_prob_rec = last_state['slate docs_log_prob_accum'].get('doc_ranks')
    log_prob_user = last_state['user response_log_prob_accum'].get('choice')
    # print(last_state['user response_log_prob'])
    objective_user = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob_user, 1)
    objective_user /= float(global_batch)
    # objective_rec = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob_rec, 1)
    # objective_rec /= float(global_batch)
    # objective = loss_object(y_true=log_prob, y_pred=y_)
  grads_rep = tape.gradient(objective_user, phi.trainable_variables+mu.trainable_variables)
  # grads_policy = tape.gradient(objective_rec, phi.trainable_variables+actor.trainable_variables)
  if optimizer:
    grads_and_vars_rep = list(zip(grads_rep, phi.trainable_variables+mu.trainable_variables))
    optimizer.apply_gradients(grads_and_vars_rep)
    # grads_and_vars_policy = list(zip(grads_policy, phi.trainable_variables+actor.trainable_variables))
    # optimizer.apply_gradients(grads_and_vars_policy)

  with tf.GradientTape() as tape:
    # traj = tf_runtime_est.trajectory(length=history_length)
    # last_state = fn_last_state(traj)
    last_state = tf_runtime_est.execute(num_steps=history_length,est=True)
    last_metric_value = last_state['metrics state'].get(metric_to_optimize)
    log_prob_rec = last_state['slate docs_log_prob_accum'].get('doc_ranks')
    objective_rec = -tf.tensordot(tf.stop_gradient(last_metric_value), log_prob_rec, 1)
    objective_rec /= float(global_batch)
  grads_policy = tape.gradient(objective_rec, phi.trainable_variables+actor.trainable_variables)
  if optimizer:
    grads_and_vars_policy = list(zip(grads_policy, phi.trainable_variables+actor.trainable_variables))
    optimizer.apply_gradients(grads_and_vars_policy)
  
  return grads_rep, objective_user, grads_policy, objective_rec, tf.reduce_mean(last_metric_value)
  # return grads_rep, objective_user, tf.reduce_mean(last_metric_value)
  


def make_runtime(variables):
  """Makes simulation + policy log-prob runtime."""
  variables = list(variables)
  slate_var = [var for var in variables if 'slate docs' == var.name]
  log_prob_slate_var = log_probability.log_prob_variables_from_direct_output(
      slate_var)
  user_var = [var for var in variables if 'user response' == var.name]
  log_prob_user_var = log_probability.log_prob_variables_from_direct_output(
      user_var)
  accumulator_slate = log_probability.log_prob_accumulator_variables(log_prob_slate_var)
  accumulator_user = log_probability.log_prob_accumulator_variables(log_prob_user_var)
  network=network_lib.Network(
          variables=list(variables) + list(log_prob_slate_var) + list(accumulator_slate)
           + list(log_prob_user_var) + list(accumulator_user))
  tf_runtime = runtime.TFRuntime(
      network=network,
      graph_compile=False)
  return tf_runtime


def make_train_step(
    tf_runtime_main,
    tf_runtime_sub,
    tf_runtime_est,
    horizon,
    history_length,
    global_batch,
    trainable_variables,
    metric_to_optimize,
    trajs_list_1,
    trajs_list_2,
    phi,
    mu,
    actor,
    critic,
    optimizer = None
):
  """Wraps a traced training step function for use in learning loops."""

  @tf.function
  def distributed_grad_and_train():
    return distributed_train_step(tf_runtime_main, tf_runtime_sub, tf_runtime_est, horizon, history_length, global_batch,
                                  trainable_variables, trajs_list_1, trajs_list_2, phi, mu, actor, critic,
                                  metric_to_optimize, optimizer)

  return distributed_grad_and_train


def run_simulation(
    num_training_steps,
    horizon,
    history_length,
    global_batch,
    learning_rate,
    simulation_variables_main,
    simulation_variables_sub,
    simulation_variables_est,
    trainable_variables,
    metric_to_optimize,
    phi,
    mu,
    actor,
    critic
):
  """Runs simulation over multiple horizon steps while learning policy vars."""
  trajs_list_1 = []
  trajs_list_2 = []

  optimizer = reset_optimizer(learning_rate)
  optimizer.build(trainable_variables)
  tf_runtime_rep = make_runtime(simulation_variables_main)
  tf_runtime_uni = make_runtime(simulation_variables_sub)
  tf_runtime_est = make_runtime(simulation_variables_est)
  train_step = make_train_step(tf_runtime_rep, tf_runtime_uni, tf_runtime_est, horizon, history_length, global_batch,
                               trainable_variables, metric_to_optimize, trajs_list_1, trajs_list_2, phi, mu,
                               actor, critic, optimizer)

  metric_list = []
  for _ in range(num_training_steps):
    _, _, _, _, last_metric = train_step()
    metric_list.append(last_metric)
    wandb.log({"acc": last_metric})

