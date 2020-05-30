# https://deeplearningcourses.com/c/cutting-edge-artificial-intelligence
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from models import Critic_gen,Actor_gen
from collections import deque
from sys import exit
from buffer import BasicBuffer_a, BasicBuffer_b
import random

# np.random.seed(0)
# tf.random.set_seed(0)


class DDPGAgent:
    
    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        # self.action_max = 1
        
        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau
        
        #Network layers
        actor_layer = [512,200,128]
        critic_layer = [1024,512,300,1]


        # Main network outputs
        self.mu = Actor_gen((3),(1),actor_layer,self.action_max)
        self.q_mu = Critic_gen((3),(1),critic_layer)

        # Target networks
        self.mu_target = Actor_gen((3),(1),actor_layer,self.action_max)
        self.q_mu_target = Critic_gen((3),(1),critic_layer)
      
        # Copying weights in,
        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())
    
        # optimizers
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        self.q_mu_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)
  
        self.replay_buffer = BasicBuffer_a(size = buffer_maxlen, obs_dim = self.obs_dim, act_dim = self.action_dim)
        
        self.q_losses = []
        
        self.mu_losses = []
        
    def get_action(self, s, noise_scale):
        a =  self.mu.predict(s.reshape(1,-1))[0]
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_max, self.action_max)

    def update(self, batch_size):
        
        
        X,A,R,X2,D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X,dtype=np.float32)
        A = np.asarray(A,dtype=np.float32)
        R = np.asarray(R,dtype=np.float32)
        X2 = np.asarray(X2,dtype=np.float32)
        
        
        Xten=tf.convert_to_tensor(X)
        

        # Updating Ze Critic
        with tf.GradientTape() as tape:
          A2 =  self.mu_target(X2)
          q_target = R + self.gamma  * self.q_mu_target([X2,A2])
          qvals = self.q_mu([X,A]) 
          q_loss = tf.reduce_mean((qvals - q_target)**2)
          grads_q = tape.gradient(q_loss,self.q_mu.trainable_variables)
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)


        #Updating ZE Actor
        with tf.GradientTape() as tape2:
          A_mu =  self.mu(X)
          Q_mu = self.q_mu([X,A_mu])
          mu_loss =  -tf.reduce_mean(Q_mu)
          grads_mu = tape2.gradient(mu_loss,self.mu.trainable_variables)
        self.mu_losses.append(mu_loss)
        self.mu_optimizer.apply_gradients(zip(grads_mu, self.mu.trainable_variables))



        # update target networks
              ## Updating both netwokrs
        # # updating q_mu network
        
        temp1 = np.array(self.q_mu_target.get_weights())
        temp2 = np.array(self.q_mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.q_mu_target.set_weights(temp3)
      

      # updating mu network
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau*temp2 + (1-self.tau)*temp1
        self.mu_target.set_weights(temp3)
        



