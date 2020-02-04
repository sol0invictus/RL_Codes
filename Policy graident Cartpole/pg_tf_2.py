
from __future__ import print_function, division
from builtins import range


import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning_bins import plot_running_avg


# so you can test different architectures
class HiddenLayer:
  def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
    self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
    self.use_bias = use_bias
    if use_bias:
      self.b = tf.Variable(np.zeros(M2).astype(np.float32))
    self.f = f

  def forward(self, X):
    if self.use_bias:
      a = tf.matmul(X, self.W) + self.b
    else:
      a = tf.matmul(X, self.W)
    return self.f(a)


# approximates pi(a | s)
class PolicyModel:
  def __init__(self, D, K, hidden_layer_sizes):
    # create the graph
    # K = number of actions
    self.layers = []
    self.vars=[]
    self.K=K
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      self.vars.append(layer.W)
      self.vars.append(layer.b)
      M1 = M2

    # final layer
    # layer = HiddenLayer(M1, K, lambda x: x, use_bias=False)
    layer = HiddenLayer(M1, K, tf.nn.softmax, use_bias=False)
    self.vars.append(layer.W)
    self.layers.append(layer)
    self.train_op = tf.keras.optimizers.Adagrad(learning_rate=0.1)
    
  def get_loss(self,X,actions, advantages):
    X = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    p_a_given_s = Z
    self.predict_op = p_a_given_s
    # print(self.predict_op)
    selected_probs = tf.math.log(tf.reduce_sum(p_a_given_s * tf.one_hot(actions, self.K),axis=[1]))
    cost = -tf.reduce_sum(advantages * selected_probs)
    return cost
    
  def get_grad(self,X,actions,advantages):
        with tf.GradientTape() as tape:
          L = self.get_loss(X,actions,advantages)
          g = tape.gradient(L, self.vars)
        return g
  
  def network_learn(self,X,actions,advantages):
    g = self.get_grad(X,actions,advantages)
    # print(self.var)
    self.train_op.apply_gradients(zip(g, self.vars))
  
  
  def partial_fit(self, X, actions, advantages):
    X = np.atleast_2d(X)
    actions = np.atleast_1d(actions)
    advantages = np.atleast_1d(advantages)
    self.network_learn(X,actions,advantages)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    p_a_given_s = Z
    return p_a_given_s.numpy()

  def sample_action(self, X):
    p = self.predict(X)[0]
    return np.random.choice(len(p), p=p)


# approximates V(s)
class ValueModel:
  def __init__(self, D, hidden_layer_sizes):
    # create the graph
    self.layers = []
    self.vars=[]
    M1 = D
    for M2 in hidden_layer_sizes:
      layer = HiddenLayer(M1, M2)
      self.layers.append(layer)
      self.vars.append(layer.W)
      self.vars.append(layer.b)
      M1 = M2

    # final layer
    layer = HiddenLayer(M1, 1, lambda x: x)
    self.vars.append(layer.W)
    self.vars.append(layer.b)
    self.layers.append(layer)
    self.train_op = tf.keras.optimizers.SGD(learning_rate=0.0001)
    


  def get_loss(self,X,Y):
    X = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1]) # the output
    self.predict_op = Y_hat

    cost = tf.reduce_sum(tf.square(Y - Y_hat))
    return cost
    
  def get_grad(self,X,Y):
        with tf.GradientTape() as tape:
          L = self.get_loss(X,Y)
          g = tape.gradient(L, self.vars)
        return g
  
  def network_learn(self,X,Y):
    g = self.get_grad(X,Y)
    # print(self.var)
    self.train_op.apply_gradients(zip(g, self.vars))

  
  def partial_fit(self, X, Y):
    X = np.atleast_2d(X)
    Y = np.atleast_1d(Y)
    self.network_learn(X,Y)

  def predict(self, X):
    X = np.atleast_2d(X)
    X = tf.convert_to_tensor(X, dtype=tf.dtypes.float32)
    Z = X
    for layer in self.layers:
      Z = layer.forward(Z)
    Y_hat = tf.reshape(Z, [-1])
    return Y_hat


def play_one_td(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  while not done and iters < 2000:
    # if we reach 2000, just quit, don't want this going forever
    # the 200 limit seems a bit early
    action = pmodel.sample_action(observation)
    prev_observation = observation
    observation, reward, done, info = env.step(action)

    # if done:
    #   reward = -200

    # update the models
    V_next = vmodel.predict(observation)[0]
    G = reward + gamma*V_next
    advantage = G - vmodel.predict(prev_observation)
    pmodel.partial_fit(prev_observation, action, advantage)
    vmodel.partial_fit(prev_observation, G)

    if reward == 1: # if we changed the reward to -200
      totalreward += reward
    iters += 1

  return totalreward



def play_one_mc(env, pmodel, vmodel, gamma):
  observation = env.reset()
  done = False
  totalreward = 0
  iters = 0

  states = []
  actions = []
  rewards = []

  reward = 0
  while not done and iters < 2000:
    action = pmodel.sample_action(observation)

    states.append(observation)
    actions.append(action)
    rewards.append(reward)

    prev_observation = observation
    observation, reward, done, info = env.step(action)

    if done:
      reward = -200

    if reward == 1: 
      totalreward += reward
    iters += 1

  # save the final (s,a,r) tuple
  action = pmodel.sample_action(observation)
  states.append(observation)
  actions.append(action)
  rewards.append(reward)

  returns = []
  advantages = []
  G = 0
  for s, r in zip(reversed(states), reversed(rewards)):
    returns.append(G)
    advantages.append(G - vmodel.predict(s)[0])
    G = r + gamma*G
  returns.reverse()
  advantages.reverse()
  # update the models

  pmodel.partial_fit(states, actions, advantages)
  vmodel.partial_fit(states, returns)

  return totalreward


def main():
  env = gym.make('CartPole-v0')
  D = env.observation_space.shape[0]
  K = env.action_space.n
  pmodel = PolicyModel(D, K, [])
  vmodel = ValueModel(D, [10])
  gamma = 0.99

  if 'monitor' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  N = 1
  totalrewards = np.empty(N)
  costs = np.empty(N)
  for n in range(N):
    totalreward = play_one_mc(env, pmodel, vmodel, gamma)
    totalrewards[n] = totalreward
    if n % 100 == 0:
      print("episode:", n, "total reward:", totalreward, "avg reward (last 100):", totalrewards[max(0, n-100):(n+1)].mean())

  print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  print("total steps:", totalrewards.sum())

  plt.plot(totalrewards)
  plt.title("Rewards")
  plt.show()

  plot_running_avg(totalrewards)


if __name__ == '__main__':
  main()

