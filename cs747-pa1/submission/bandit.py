import numpy as np
import matplotlib
import os
import sys
from utils import kl_div, beta_pdf, kl_div_array


class MultiArmedBandit(object):
    """A Multi-Armed Bandit"""
    def __init__(self,p_list):
        """
        Args:
            p_list (List[float]): Mean reward for each arm
        """
        self.n = len(p_list)
        self.p_list = np.array(p_list)
        self.succ = np.zeros(self.n)
        self.fail = np.zeros(self.n)
        self.hint = None
        self.hint_weights = None

    def set_hint(self,hint):
        """Set the hint for Thompson Sampling with hint

        Args:
            hint (numpy.ndarray): Sorted list of true means
        """
        self.hint = hint
        self.hint_weights = np.ones((self.n,self.n))

    def pull(self,i):
        """Pulling an arm

        Args:
            i (int): Index of the arm to be pulled

        Returns:
            float: Reward for the pull
        """
        reward = np.random.binomial(size=1,n=1,p=self.p_list[i])[0]
        if reward == 0:
            self.fail[i]+=1
            if self.hint is not None:
                self.hint_weights[i] = self.hint_weights[i] * (1-self.hint) * (self.succ[i]+self.fail[i]+1)/self.fail[i] 

        else:
            self.succ[i]+=1
            if self.hint is not None:
                self.hint_weights[i] = self.hint_weights[i] * self.hint * (self.succ[i]+self.fail[i]+1)/self.succ[i]
        return reward

    
    def get_max_emperical_mean(self):
        """Returns the arm with highest emperical mean

        Returns:
            int: Arm index
        """
        emperical_means = self.succ/(self.succ+self.fail)
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
        emperical_means = self.succ/(self.succ+self.fail)
        ucbs = emperical_means + np.sqrt(2*np.log(time)/(self.succ+self.fail))
        return np.argmax(ucbs)

    def get_max_kl_ucb(self,time,c,precision):
        """Returns the arm with maximum kl_ucb

        Args:
            time (int): Time in the experiment
            c (int): Value of c in the KL-UCB equation
            precision(float): Precision for Binary Search 

        Returns:
            int: Arm index
        """
        emperical_means = self.succ/(self.succ+self.fail)
        low = emperical_means.copy()
        high = np.ones(self.n)
        rhs = np.log(time)
        mid = np.ones(self.n)
        lhs = np.ones(self.n)
        while not all(np.abs(low-high)< precision):
            cond = (np.abs(high-low)>=precision)
            mid[cond] = (low[cond]+high[cond])/2
            lhs[cond] = (self.succ[cond] + self.fail[cond])*kl_div_array(emperical_means[cond],mid[cond])
            low[ (lhs<=rhs) * cond ] = mid[(lhs<=rhs) * cond]
            high[(lhs>rhs) * cond ] = mid[(lhs>rhs) * cond]
        return np.argmax(low)

        # emp_mean = self.get_emperical_mean()
        # low = emp_mean
        # high = 1
        # while(True):
        #     mid = (low+high)/2
        #     if np.abs(low-high) < 1e-2:
        #         break
        #     lhs = kl_div(emp_mean,mid)*(self.succ+self.fail)
        #     rhs = np.log(time) + 3*np.log(np.log(time))
        #     if lhs <= rhs:
        #         low = mid
        #     else:
        #         high = mid
            
        # return mid
        


    def get_max_thompson_sample(self):
        """Returns the arm with maximum sampled mean using Thompson Sampling

        Returns:
            int: Arm index
        """
        thompson_samples = np.random.beta(self.succ+1,self.fail+1)
        return np.argmax(thompson_samples)

    def get_max_thompson_hint_sample(self):
        """Returns the arm with maximum sampled mean using Thompson Sampling with hint

        Args:
            hint(numpy.ndarray): Mean rewards of the arms

        Returns:
            int: Arm index
        """
        max_hint_index = np.argmax(self.hint)
        return np.argmax(self.hint_weights[:,max_hint_index])


    
        