import numpy as np
import matplotlib
import os
import sys

class BanditArm(object):
    """A single Bandit Arm"""
    def __init__(self,prob):
        """
        Args:
            prob (float):  Mean reward for the bandit
        """
        self.prob = prob
        self.succ = 0
        self.fail = 0

    def pull(self):
        """Pulling the arm

        Returns:
            int: Reward obtained on pulling the arm
        """
        reward = np.random.binomial(size=1,n=1,p=self.prob)[0]
        if reward == 0:
            self.fail+=1
        else:
            self.succ+=1
        return reward

    def get_emperical_mean(self):
        """Emperical mean of the arm

        Returns:
            float: Emperical mean
        """
        return self.succ/(self.succ+self.fail)


class MultiArmedBandit(object):
    """A Multi-Armed Bandit"""
    def __init__(self,p_list):
        """
        Args:
            p_list (List[float]): Mean reward for each arm
        """
        self.p_list = np.array(p_list)
        self.arms = []
        for prob in p_list:
            self.arms.append(BanditArm(prob))
    
    def get_emperical_means(self):
        """Returns the emperical mean for each arm

        Returns:
            numpy.ndarray: List of emperical means
        """
        emperical_means = []
        for arm in self.arms:
            emperical_means.append(arm.get_emperical_mean())
        return np.array(emperical_means)


    def max_exp_rew(self,horizon):
        """Returns maximum expected cumulative reward

        Args:
            horizon (int): Horizon duration

        Returns:
            float: Maximum expected cumulative reward
        """
        return max(self.p_list)*horizon


    
        