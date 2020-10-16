import numpy as np
import matplotlib
import os
import sys
from utils import kl_div, beta_pdf, kl_div_array, parser
from algorithms import *


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
        self.hint_weights = np.ones((self.n,self.n))/self.n

    def pull(self,i):
        """Pulling an arm

        Args:
            i (int): Index of the arm to be pulled

        Returns:
            float: Reward for the pull
        """
        reward = 1 if np.random.uniform(0,1)<=self.p_list[i] else 0
        if reward == 0:
            self.fail[i]+=1
            if self.hint is not None:
                self.hint_weights[i] = self.hint_weights[i] * (1-self.hint)
                self.hint_weights[i] /= np.sum(self.hint_weights[i])

        else:
            self.succ[i]+=1
            if self.hint is not None:
                self.hint_weights[i] = self.hint_weights[i]*self.hint 
                self.hint_weights[i] /= np.sum(self.hint_weights[i])
        
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
        rhs = np.log(time) + c*np.log(np.log(time))
        mid = np.ones(self.n)
        lhs = np.ones(self.n)
        while not all(np.abs(low-high)< precision):
            cond = (np.abs(high-low)>=precision)
            mid[cond] = (low[cond]+high[cond])/2
            lhs[cond] = (self.succ[cond] + self.fail[cond])*kl_div_array(emperical_means[cond],mid[cond])
            low[ (lhs<=rhs) * cond ] = mid[(lhs<=rhs) * cond]
            high[(lhs>rhs) * cond ] = mid[(lhs>rhs) * cond]
        return np.argmax(low)


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

if __name__ == "__main__":
    args = parser()
    p_list = []
    with open(args.instance,'r') as f:
        lines = f.readlines()
        for line in lines:
            try:
                p_list.append(float(line.replace("\n","").strip()))
            except:
                continue
    mab = MultiArmedBandit(p_list)
    np.random.seed(args.randomSeed)
    if args.algorithm == 'epsilon-greedy':
        regrets = epsilon_greedy(mab,args.epsilon,[args.horizon])
    elif args.algorithm == 'ucb':
        regrets = ucb(mab,[args.horizon])
    elif args.algorithm == 'kl-ucb':
        regrets = kl_ucb(mab,0,1e-3,[args.horizon])
    elif args.algorithm == 'thompson-sampling':
        regrets = thompson_sampling(mab,[args.horizon])
    elif args.algorithm == 'thompson-sampling-with-hint':
        regrets = thompson_sampling_with_hint(mab,[args.horizon])
    
    output = args.instance+", "+args.algorithm+", "+str(args.randomSeed)+", "+str(args.epsilon)+", "+str(args.horizon)+", "+str(regrets[args.horizon])+"\n"
    print(output)

    
        