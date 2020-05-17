# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from datetime import datetime





# simple feedforward neural net
# TF2
def ANN2(input_shape,layer_sizes, hidden_activation='relu', output_activation=None):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=input_shape))    
    for h in layer_sizes[:-1]:
        x = model.add(tf.keras.layers.Dense(units=h, activation='relu'))
    model.add(tf.keras.layers.Dense(units=layer_sizes[-1], activation=output_activation))
    return model
      


# get all variables within a scope
def get_vars(scope):
  return [x for x in tf.global_variables() if scope in x.name]


### Create both the actor and critic networks at once ###
### Q(s, mu(s)) returns the maximum Q for a given state s ###
def CreateNetworks(
    s, a,
    num_actions,
    action_max,
    hidden_sizes=(300,),
    hidden_activation=tf.nn.relu, 
    output_activation=tf.tanh):

  with tf.variable_scope('mu'):
    mu = action_max * ANN(s, list(hidden_sizes)+[num_actions], hidden_activation, output_activation)
  with tf.variable_scope('q'):
    input_ = tf.concat([s, a], axis=-1) # (state, action)
    q = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
  with tf.variable_scope('q', reuse=True):
    # reuse is True, so it reuses the weights from the previously defined Q network
    input_ = tf.concat([s, mu], axis=-1) # (state, mu(state))
    q_mu = tf.squeeze(ANN(input_, list(hidden_sizes)+[1], hidden_activation, None), axis=1)
  return mu, q, q_mu


### The experience replay memory ###
class ReplayBuffer:
  def __init__(self, obs_dim, act_dim, size):
    self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
    self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
    self.rews_buf = np.zeros(size, dtype=np.float32)
    self.done_buf = np.zeros(size, dtype=np.float32)
    self.ptr, self.size, self.max_size = 0, 0, size

  def store(self, obs, act, rew, next_obs, done):
    self.obs1_buf[self.ptr] = obs
    self.obs2_buf[self.ptr] = next_obs
    self.acts_buf[self.ptr] = act
    self.rews_buf[self.ptr] = rew
    self.done_buf[self.ptr] = done
    self.ptr = (self.ptr+1) % self.max_size
    self.size = min(self.size+1, self.max_size)

  def sample_batch(self, batch_size=32):
    idxs = np.random.randint(0, self.size, size=batch_size)
    return dict(s=self.obs1_buf[idxs],
                s2=self.obs2_buf[idxs],
                a=self.acts_buf[idxs],
                r=self.rews_buf[idxs],
                d=self.done_buf[idxs])


### Implement the DDPG algorithm ###
def ddpg(
    env_fn,
    ac_kwargs=dict(),
    seed=0,
    save_folder=None,
    num_train_episodes=100,
    test_agent_every=25,
    replay_size=int(1e6),
    gamma=0.99, 
    decay=0.995,
    mu_lr=1e-3,
    q_lr=1e-3,
    batch_size=100,
    start_steps=10000, 
    action_noise=0.1,
    max_episode_length=1000):

  tf.random.set_seed(seed)
  np.random.seed(seed)

  env, test_env = env_fn(), env_fn()

  # comment out this line if you don't want to record a video of the agent
  # if save_folder is not None:
  #   test_env = gym.wrappers.Monitor(test_env, save_folder)

  # get size of state space and action space
  num_states = env.observation_space.shape[0]
  num_actions = env.action_space.shape[0]

  # Maximum value of action
  # Assumes both low and high values are the same
  # Assumes all actions have the same bounds
  # May NOT be the case for all environments
  action_max = env.action_space.high[0]

  # Create Tensorflow placeholders (neural network inputs)
  # X = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # state
  # A = tf.placeholder(dtype=tf.float32, shape=(None, num_actions)) # action
  # X2 = tf.placeholder(dtype=tf.float32, shape=(None, num_states)) # next state
  # R = tf.placeholder(dtype=tf.float32, shape=(None,)) # reward
  # D = tf.placeholder(dtype=tf.float32, shape=(None,)) # done



  # Main Network Creation
  X_shape = (num_states)
  QA_shape = (num_states + num_actions)
  hidden_sizes=(300,)
  hidden_activation='relu' 
  output_activation='tanh'
  # Main network outputs
  mu = ANN2(X_shape,list(hidden_sizes)+[num_actions], hidden_activation='relu', output_activation=None)
  # q = ANN2(QA_shape, list(hidden_sizes)+[1], hidden_activation, None) 
  q_mu = ANN2(QA_shape, list(hidden_sizes)+[1], hidden_activation, None)

  # Target networks
  mu_target = ANN2(X_shape,list(hidden_sizes)+[num_actions], hidden_activation='relu', output_activation=None)
  
  q_mu_target = ANN2(QA_shape, list(hidden_sizes)+[1], hidden_activation, None)
 
  # Experience replay memory
  replay_buffer = ReplayBuffer(obs_dim=num_states, act_dim=num_actions, size=replay_size)


  # Target value for the Q-network loss
  # We use stop_gradient to tell Tensorflow not to differentiate
  # q_mu_targ wrt any params
  # i.e. consider q_mu_targ values constant


  # Train each network separately
  mu_optimizer =tf.keras.optimizers.Adam(learning_rate=mu_lr)
  q_optimizer = tf.keras.optimizers.Adam(learning_rate=q_lr)
  
  # mu_train_op = mu_optimizer.minimize(mu_loss, var_list=get_vars('main/mu'))
  # q_train_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

  # Use soft updates to update the target networks
  # target_update = tf.group(
  #   [tf.assign(v_targ, decay*v_targ + (1 - decay)*v_main)
  #     for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
  #   ]
  # )

  # Copy main network params to target networks
  # target_init = tf.group(
  #   [tf.assign(v_targ, v_main)
  #     for v_main, v_targ in zip(get_vars('main'), get_vars('target'))
  #   ]
  # )

  # boilerplate (and copy to the target networks!)
  # sess.run(target_init)

  def get_action(s, noise_scale):
    a = action_max * mu.predict(s.reshape(1,-1))[0]
    a += noise_scale * np.random.randn(num_actions)
    return np.clip(a, -action_max, action_max)

  from sys import exit
  print(num_actions)
  # exit(0)
  # Main loop: play episode and train
  returns = []
  q_losses = []
  mu_losses = []
  num_steps = 0
  for i_episode in range(num_train_episodes):

    # reset env
    s, episode_return, episode_length, d = env.reset(), 0, 0, False

    while not (d or (episode_length == max_episode_length)):
      # For the first `start_steps` steps, use randomly sampled actions
      # in order to encourage exploration.
      
      if num_steps > start_steps:
        a = get_action(s, action_noise)
      else:
        a = env.action_space.sample()

      # Keep track of the number of steps done
      num_steps += 1
      if num_steps == start_steps:
        print("USING AGENT ACTIONS NOW")

      # Step the env
      s2, r, d, _ = env.step(a)
      episode_return += r
      episode_length += 1

      # Ignore the "done" signal if it comes from hitting the time
      # horizon (that is, when it's an artificial terminal signal
      # that isn't based on the agent's state)
      d_store = False if episode_length == max_episode_length else d

      # Store experience to replay buffer
      replay_buffer.store(s, a, r, s2, d_store)

      # Assign next state to be the current state on the next round
      s = s2

    # Perform the updates
    for _ in range(episode_length):
      batch = replay_buffer.sample_batch(batch_size)
      X = batch['s']
      X2 = batch['s2']
      A = batch['a']
      R = batch['r']
      D = batch['d']
      # print(X2.shape)
      Xten=tf.convert_to_tensor(X)

      # Q network update
      # Note: plot the Q loss if you want
      with tf.GradientTape() as tape2:
        boom = mu(X)
        tempworld = tf.keras.layers.concatenate([Xten,boom],axis=1)
        temp2 = q_mu(tempworld)
        mu_loss = -tf.reduce_mean(temp2)
        # mu_loss = -tf.reduce_mean(tempworld)
        # mu_loss = mu(X)
        print(mu_loss)
        # exit(0)
        grads_mu = tape2.gradient(mu_loss,mu.trainable_variables)
        # print(grads_mu)
      # print(mu.trainable_variables)
      # exit(0)
      mu_optimizer.apply_gradients(zip(grads_mu, mu.trainable_variables))
      mu_losses.append(mu_loss)
      
      with tf.GradientTape() as tape:
        next_a = action_max * mu_target(X2)
        temp = np.concatenate((X2,next_a),axis=1)
        q_target = R + gamma * (1 - D) * q_mu_target(temp)
        # print('here')
        temp2 = np.concatenate((X,A),axis=1)
        # DDPG losses
        # mu_loss = -tf.reduce_mean(q_mu(np.concatenate((X,mu(X)),axis=1)))
        qvals = q_mu(temp2) 
        q_loss = tf.reduce_mean((qvals - q_target)**2)
        print(q_loss)
        # exit(0)
        grads_q = tape.gradient(q_loss,q_mu.trainable_variables)
        # grads_mu = tape.gradient(mu_loss,mu.trainable_variables)
        # print(grads)
        # exit(0)
      q_optimizer.apply_gradients(zip(grads_q, q_mu.trainable_variables))
      # mu_optimizer.apply_gradients(zip(mu_loss, mu.trainable_variables))
      q_losses.append(q_loss)
      # mu_losses.append(mu_loss)

      # Policy update
      # (And target networks update)
      # Note: plot the mu loss if you want
      # mul, _, _ = sess.run([mu_loss, mu_train_op, target_update], feed_dict)
      # mu_losses.append(mul)

    print("Episode:", i_episode + 1, "Return:", episode_return, 'episode_length:', episode_length)
    returns.append(episode_return)

    # Test the agent
    if i_episode > 0 and i_episode % test_agent_every == 0:
      test_agent()

  # on Mac, plotting results in an error, so just save the results for later
  # if you're not on Mac, feel free to uncomment the below lines
  np.savez('ddpg_results.npz', train=returns, test=test_returns, q_losses=q_losses, mu_losses=mu_losses)

  # plt.plot(returns)
  # plt.plot(smooth(np.array(returns)))
  # plt.title("Train returns")
  # plt.show()

  # plt.plot(test_returns)
  # plt.plot(smooth(np.array(test_returns)))
  # plt.title("Test returns")
  # plt.show()

  # plt.plot(q_losses)
  # plt.title('q_losses')
  # plt.show()

  # plt.plot(mu_losses)
  # plt.title('mu_losses')
  # plt.show()


def smooth(x):
  # last 100
  n = len(x)
  y = np.zeros(n)
  for i in range(n):
    start = max(0, i - 99)
    y[i] = float(x[start:(i+1)].sum()) / (i - start + 1)
  return y


if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
  parser.add_argument('--env', type=str, default='Pendulum-v0')
  parser.add_argument('--hidden_layer_sizes', type=int, default=300)
  parser.add_argument('--num_layers', type=int, default=1)
  parser.add_argument('--gamma', type=float, default=0.99)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--num_train_episodes', type=int, default=200)
  parser.add_argument('--save_folder', type=str, default='ddpg_monitor')
  args = parser.parse_args()


  ddpg(
    lambda : gym.make(args.env),
    ac_kwargs=dict(hidden_sizes=[args.hidden_layer_sizes]*args.num_layers),
    gamma=args.gamma,
    seed=args.seed,
    save_folder=args.save_folder,
    num_train_episodes=args.num_train_episodes,
  )
