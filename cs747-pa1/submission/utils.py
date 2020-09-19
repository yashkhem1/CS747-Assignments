import numpy as np
import matplotlib
import math
import argparse

def beta_pdf(a,b,x):
    """Beta probabilities for the numbers in x

    Args:
        a (float): alpha
        b (float): beta
        x (numpy.ndarray): List of numbers for which probability is to be calculated

    Returns:
        numpy.ndarray: List of Beta probabilities for the given numbers
    """ 
    # return x**(a-1) * (1-x)**(b-1) * math.gamma(a+b) / math.gamma(a) / math.gamma(b)
    ln_ans = (a-1) * np.log(x) + (b-1) * np.log(1-x) + np.sum(np.log(np.arange(1,a+b))) - np.sum(np.log(np.arange(1,a))) - np.sum(np.log(np.arange(1,b))) 
    return np.exp(ln_ans)


def kl_div(x,y):
    """KL Divergence between 2 bernoulli  distributions

    Args:
        x (float): mean of first dist.
        y (float): mean of second dist.

    Returns:
        float: KL Divergence
    """
    if y == 1:
        return np.inf
    if x == 0:
        return np.log(1/(1-y))
    else:
        return x*np.log(x/y) + (1-x)*np.log((1-x)/(1-y))

def kl_div_array(x,y):
    """Bernoulli KL Divergence between 2 lists

    Args:
        x (numpy.ndarray): List of means of first dist.
        y (numpy.ndarray): List of means of second dist.

    Returns:
        numpy.ndarray: List of KL Divergences
    """
    kl_div = np.zeros(len(x))
    kl_div[y==1] = np.inf
    cond1 = (x==0) * (y!=1)
    kl_div[cond1] = np.log(1/(1-y[cond1]))
    cond2 = (x!=0) * (y!=1)
    kl_div[cond2] = x[cond2]*np.log(x[cond2]/y[cond2]) + (1-x[cond2])*np.log((1-x[cond2])/(1-y[cond2]))
    return kl_div
    

def parser():
    """Command Line Parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--instance',type=str,help='Path to the bandit instance file')
    parser.add_argument('--algorithm',type=str,help='Algorithm for pulling the bandits', choices=['epsilon-greedy','ucb','kl-ucb',
                                                                                        'thompson-sampling','thompson-sampling-with-hint'])
    parser.add_argument('--epsilon',type=float,help='Epsilon for epsilon-greedy')
    parser.add_argument('--randomSeed',type=int,help='Seed for initializing Random Generator')
    parser.add_argument('--horizon', type=int, help='Horizon till which experiment is to be performed')
    
