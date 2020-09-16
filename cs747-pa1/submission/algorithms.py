import numpy as np
import matplotlib
import os
import sys
from bandit import MultiArmedBandit, BanditArm

def epsilon_greedy(mab, epsilon, horizons):
    """Epsilon Greedy  Algorithm

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit instance
        epsilon (float): Epsilon
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
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
            arm_index = mab.get_max_emperical_mean()
            rew = mab.arms[arm_index].pull()
            cum_reward += rew

        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def ucb(mab,horizons):
    num_arms = len(mab.arms)
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for arm in mab.arms:
        rew = arm.pull()
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_ucb(i+1)
        rew = mab.arms[arm_index].pull()
        cum_reward += rew
    if i+1 in horizons:
        regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets


def kl_ucb(mab, horizons):
    num_arms = len(mab.arms)
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for arm in mab.arms:
        rew = arm.pull()
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_kl_ucb(i+1)
        rew = mab.arms[arm_index].pull()
        cum_reward += rew
    if i+1 in horizons:
        regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def thompson_sampling(mab, horizons):
    num_arms = len(mab.arms)
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for arm in mab.arms:
        rew = arm.pull()
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_thompson_sample()
        rew = mab.arms[arm_index].pull()
        cum_reward += rew
    if i+1 in horizons:
        regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def thompson_sampling_with_hints(mab, horizons):
    num_arms = len(mab.arms)
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    hint = np.sort(mab.p_list)
    for arm in mab.arms:
        rew = arm.pull()
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_thompson_hint_sample(hint)
        rew = mab.arms[arm_index].pull()
        cum_reward += rew
    if i+1 in horizons:
        regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

if __name__ == '__main__':
    np.random.seed(14)
    mab = MultiArmedBandit([0.3,0.5,0.7])
    print(thompson_sampling_with_hints(mab,[102400]))

    