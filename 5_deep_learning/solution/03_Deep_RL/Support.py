# Copyright 2017 Mark Pfeiffer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Fadri Furrer, ASL, ETH Zurich, Switzerland
# Copyright 2017 Renaud Dubé, ASL, ETH Zurich, Switzerland

import numpy as np

def discountedReward(reward_array, gamma):
    """
    Compute discounted reward over time series.

    Each timestep will give a reward of 1.
    This function leaves the final reward (1) unchanged but increases the
    shorter term reward.
    """
    discounted_reward = np.zeros([len(reward_array), 1])
    running_reward = 0
    for i in reversed(range(0, len(reward_array))):
        running_reward = running_reward * gamma + reward_array[i]
        discounted_reward[i] = running_reward
    return discounted_reward
