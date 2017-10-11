####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import tensorflow as tf
import time
import numpy as np
import gym
import pylab as pl
import progressbar

from NNModel import *

# Set up the simulation environment
env = gym.make('CartPole-v0')
env.reset()
env.render()
env.close()

# Params 
num_test_runs = 100
max_num_steps_per_episode = 5000
num_steps_per_episode = []
do_visualize = False

# Reset TF
tf.reset_default_graph()

# Model definition
model = NNModel()
saver = tf.train.Saver()
checkpoint_path = 'model'

p_bar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()], 
                                maxval=num_test_runs).start()
with tf.Session() as sess: 
  # Restore model
  print('Loading session from "{}"'.format(checkpoint_path))
  ckpt = tf.train.get_checkpoint_state(checkpoint_path)
  print('Restoring model {}'.format(ckpt.model_checkpoint_path))
  saver.restore(sess, ckpt.model_checkpoint_path)
  

  # Visualize once after optimization
  for i in range(num_test_runs):
    observation = env.reset()
    done= False
    reward_episode = 0
    while not done and reward_episode < max_num_steps_per_episode:
      # Compute action 
      x = np.reshape(observation, [1, model.input_dim])
      network_output = sess.run(model.action_probability, feed_dict={model.input: x})
      # now don't take a stochastic but deterministic action
      action = 1 if network_output > 0.5 else 0 
      # Simulate
      observation, reward, done, _ = env.step(action)
      reward_episode += reward
      if do_visualize:
        env.render()
    num_steps_per_episode.append(reward_episode)
    p_bar.update(i)

p_bar.finish()  
env.close()

print('### FINAL SCORE ###')
print('Avg. steps per episode: {}'.format(np.mean(num_steps_per_episode)))
