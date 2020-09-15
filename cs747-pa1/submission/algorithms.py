import numpy as np
import matplotlib
import os
import sys
from bandit import MultiArmedBandit, BanditArm

def epsilon_greedy(mab, epsilon, seed, horizons):
    np.random.seed(seed)
    num_arms = len(mab.arms)
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for arm in mab.arms:
        rew = arm.pull()
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        frac = np.random.uniform(0,1)
        if frac < epsilon:
            arm_index = np.random.choice(np.arange(num_arms),1)[0]
            rew = mab.arms[arm_index].pull()
            cum_reward += rew
        
        else:
            emperical_means = mab.get_emperical_means
            arm_index = np.argmax(emperical_means)
            rew = mab.arms[arm_index].pull()
            cum_reward += rew

        if i+1 in horizons:
            print(cum_reward)
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets


if __name__ == '__main__':
    mab = MultiArmedBandit([0.3,0.5,0.7])
    print(epsilon_greedy(mab,0.4,1,[10000]))

    