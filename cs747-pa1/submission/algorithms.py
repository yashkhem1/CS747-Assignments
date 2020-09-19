import numpy as np
import matplotlib
import os
import sys
from bandit import MultiArmedBandit

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
        arm_index = mab.get_max_ucb(i+1)
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets


def kl_ucb(mab, horizons):
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
    for i in range(num_arms):
        rew = mab.pull(i)
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_kl_ucb(i+1,0,1e-3)
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
    num_arms = mab.n
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    for i in range(num_arms):
        rew = mab.pull(i)
        cum_reward += rew
    for i in range(num_arms,max_horizon):
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
    num_arms = mab.n
    cum_reward = 0
    regrets = {}
    max_horizon = max(horizons)
    hint = np.sort(mab.p_list)
    mab.set_hint(hint)
    for i in range(num_arms):
        rew = mab.pull(i)
        cum_reward += rew
    for i in range(num_arms,max_horizon):
        arm_index = mab.get_max_thompson_hint_sample()
        rew = mab.pull(arm_index)
        cum_reward += rew
        if i+1 in horizons:
            regrets[i+1] = mab.max_exp_rew(i+1) - cum_reward

    return regrets

if __name__ == '__main__':
    total_regrets = np.zeros(6)
    for seed in range(50):
        print(seed)
        np.random.seed(seed)
        # mab = MultiArmedBandit([0.4,0.3,0.5,0.2,0.1])
        mab = MultiArmedBandit([0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49])
        regrets = thompson_sampling_with_hint(mab,[100,400,1600,6400,25600,102400])
        total_regrets += np.array(list(regrets.values()))
    print(total_regrets/50)

    