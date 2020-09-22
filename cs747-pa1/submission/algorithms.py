import numpy as np
import matplotlib
import os
import sys
# from bandit import MultiArmedBandit

def epsilon_greedy(mab, epsilon, horizons):
    """Epsilon Greedy  Algorithm

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit instance
        epsilon (float): Epsilon
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
    num_arms = mab.n
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for i in range(num_arms):
        rew = mab.pull(i)
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        frac = np.random.uniform(0,1)
        if frac < epsilon:
            arm_index = np.random.choice(np.arange(num_arms),1)[0]
            rew = mab.pull(arm_index)
            cum_reward += rew
        
        else:
            arm_index = mab.get_max_emperical_mean()
            rew = mab.pull(arm_index)
            cum_reward += rew

        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def ucb(mab,horizons):
    """UCB Algorithm

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit Instance
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
    num_arms = mab.n
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for i in range(num_arms):
        rew = mab.pull(i)
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_ucb(i)
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets


def kl_ucb(mab, c, precision, horizons):
    """KL-UCB Algorithm

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit Instance
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
    num_arms = mab.n
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    init_pulls = 0
    while(True):
        for i in range(num_arms):
            rew = mab.pull(i)
            cum_reward += rew
            init_pulls+=1
        if np.log(init_pulls) + c*np.log(np.log(init_pulls)) >=0:
            break   
    for i in range(init_pulls,max_horizon):
        arm_index = mab.get_max_kl_ucb(i,c,precision)
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def thompson_sampling(mab, horizons):
    """Thompson Sampling Algorithm

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit Instance
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons) 
    for i in range(max_horizon):
        arm_index = mab.get_max_thompson_sample()
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

def thompson_sampling_with_hint(mab, horizons):
    """Thompson Sampling Algorithm with hint

    Args:
        mab (MultiArmedBandit): Multi-Armed Bandit Instance
        horizons (List[int]): Horizon values for which regret is to be calculated

    Returns:
        dict: Dictionary that maps horizon values to cumulative regret
    """
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    hint = np.sort(mab.p_list)
    mab.set_hint(hint)
    for i in range(max_horizon):
        arm_index = mab.get_max_thompson_hint_sample()
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets


    