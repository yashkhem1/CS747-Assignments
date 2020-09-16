import numpy as np
import matplotlib
import os
import sys
from utils import kl_div, beta_pdf

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

    def get_ucb(self,time):
        """Returns the ucb value for the arm

        Args:
            time (int): Time in the experiment

        Returns:
            float: ucb value for the arm
        """
        emp_mean = self.get_emperical_mean()
        return emp_mean + np.sqrt(2*np.log(time)/(self.succ+self.fail))

    def get_kl_ucb(self,time):
        """Returns the kl-ucb value for the arm

        Args:
            time (int): Time in the experiment

        Returns:
            float: kl-ucb value for the arm
        """
        emp_mean = self.get_emperical_mean()
        low = emp_mean
        high = 1
        while(True):
            mid = (low+high)/2
            if np.abs(low-high) < 1e-2:
                break
            lhs = kl_div(emp_mean,mid)*(self.succ+self.fail)
            rhs = np.log(time) + 3*np.log(np.log(time))
            if lhs <= rhs:
                low = mid
            else:
                high = mid
            
        return mid

    def get_thompson_sample(self):
        """Draw Thompson sample from the arm

        Returns:
            float: Thompson sample for mean reward
        """
        return np.random.beta(self.succ+1,self.fail+1)

    def get_thompson_hint_sample(self,hints):
        """Draw Thompson sample from the arm using hints 

        Args:
            hints (numpy.ndarray): Mean rewards for the arms in sorted order 

        Returns:
            float: Sample for mean reward
        """
        # x = np.random.beta(self.succ+1,self.fail+1)
        # return hints[np.argmin(np.abs(hints-x))]
        hint_weights = beta_pdf(self.succ+1, self.fail+1, hints)
        hint_weights = hint_weights/np.sum(hint_weights)
        return np.random.choice(hints,p=hint_weights)


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
    
    def get_max_emperical_mean(self):
        """Returns the arm with highest emperical mean

        Returns:
            int: Arm index
        """
        emperical_means = []
        for arm in self.arms:
            emperical_means.append(arm.get_emperical_mean())
        return np.argmax(emperical_means)


    def max_exp_rew(self,horizon):
        """Returns maximum expected cumulative reward

        Args:
            horizon (int): Horizon duration

        Returns:
            float: Maximum expected cumulative reward
        """
        return max(self.p_list)*horizon

    def get_max_ucb(self,time):
        """Returns the arm with maximum ucb

        Args:
            time (time): Time in the experiment

        Returns:
            int: Arm index
        """
        ucbs = []
        for arm in self.arms:
            ucbs.append(arm.get_ucb(time))
        return np.argmax(ucbs)

    def get_max_kl_ucb(self,time):
        """Returns the arm with maximum kl_ucb

        Args:
            time (time): Time in the experiment

        Returns:
            int: Arm index
        """
        kl_ucbs = []
        for arm in self.arms:
            kl_ucbs.append(arm.get_kl_ucb(time))
        return np.argmax(kl_ucbs)


    def get_max_thompson_sample(self):
        """Returns the arm with maximum sampled mean using Thompson Sampling

        Returns:
            int: Arm index
        """
        thompson_samples = []
        for arm in self.arms:
            thompson_samples.append(arm.get_thompson_sample())
        return np.argmax(thompson_samples)

    def get_max_thompson_hint_sample(self,hint):
        """Returns the arm with maximum sampled mean using Thompson Sampling with hint

        Args:
            hint(numpy.ndarray): Mean rewards of the arms

        Returns:
            int: Arm index
        """
        thompson_hint_samples = []
        for arm in self.arms:
            thompson_hint_samples.append(arm.get_thompson_hint_sample(hint))
        return np.argmax(thompson_hint_samples)


    
        