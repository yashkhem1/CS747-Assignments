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
    return x**(a-1) * (1-x)**(b-1) * math.gamma(a+b) / math.gamma(a) / math.gamma(b)

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
    
