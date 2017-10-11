####################################
# Author: Mark Pfeiffer            #
# Date created: 30.05.2017         #
#                                  #
# Date last changed: 11.10.2017    #
# Changed by: Mark Pfeiffer        #
####################################

import numpy as np


def discountedReward(reward_array, gamma):
  '''
  Compute discounted reward over time series. 
  Each timestep will give a reward of 1. 
  This function leaves the final reward (1) unchanged but increases the shorter term reward.
  '''
  discounted_reward = np.zeros([len(reward_array), 1])
  running_reward = 0
  for i in reversed(range(0, len(reward_array))):
    running_reward = running_reward * gamma + reward_array[i]
    discounted_reward[i] = running_reward
  return discounted_reward
