import numpy as np
import os
import sys
import random
import argparse
from windy_gridworld import WindyGridworld

class Agent(object):
    def __init__(self,algorithm,mdp:WindyGridworld,seed,num_timesteps,epsilon,lr):
        self.algorithm = algorithm
        self.mdp = mdp
        self.num_timesteps = num_timesteps
        self.epsilon = epsilon
        self.lr = lr
        self.start_state = mdp.start_state
        self.end_state = mdp.end_state
        self.num_states = mdp.num_states
        self.eight_moves = mdp.eight_moves
        if self.eight_moves:
            self.num_actions = 8
        else:
            self.num_actions = 4
        self.q_matrix = np.zeros((self.num_states,self.num_actions))
        self.seed = seed
        self.timesteps = []
        self.episodes = []
    

    def sarsa(self):
        timestep = 0
        episode = 0
        curr_state = self.start_state
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0,self.num_actions)
        else:
            action = np.argmax(self.q_matrix[curr_state])
        while timestep < self.num_timesteps:
            self.timesteps.append(timestep)
            self.episodes.append(episode)
            next_state,reward = self.mdp.step(action)
            if next_state == self.end_state:
                self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(reward - self.q_matrix[curr_state,action])
                episode += 1
                self.mdp.reset()
                curr_state = self.start_state
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0,self.num_actions)
                else:
                    action = np.argmax(self.q_matrix[curr_state])
            else:
                if np.random.rand() < self.epsilon:
                    next_action = np.random.randint(0,self.num_actions)
                else:
                    next_action = np.argmax(self.q_matrix[next_state])
                self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(reward + self.q_matrix[next_state,next_action] - self.q_matrix[curr_state,action])
                curr_state = next_state
                action = next_action
            timestep+=1


    def expected_sarsa(self):
        timestep = 0
        episode = 0
        curr_state = self.start_state
        while timestep < self.num_timesteps:
            self.timesteps.append(timestep)
            self.episodes.append(episode)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0,self.num_actions)
            else:
                action = np.argmax(self.q_matrix[curr_state])
            next_state,reward = self.mdp.step(action)
            expectation = (1-self.epsilon)*np.max(self.q_matrix[next_state]) + self.epsilon*np.mean(self.q_matrix[next_state])
            target = reward + expectation
            self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(target - self.q_matrix[curr_state,action])
            if next_state == self.end_state:
                episode += 1
                self.mdp.reset()
                curr_state = self.start_state
            else:
                curr_state = next_state
            timestep+=1

        

    def q_learning(self):
        timestep = 0
        episode = 0
        curr_state = self.start_state
        while timestep < self.num_timesteps:
            self.timesteps.append(timestep)
            self.episodes.append(episode)
            if np.random.rand() < self.epsilon:
                action = np.random.randint(0,self.num_actions)
            else:
                action = np.argmax(self.q_matrix[curr_state])
            next_state,reward = self.mdp.step(action)
            target = reward + np.max(self.q_matrix[next_state])
            self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(target - self.q_matrix[curr_state,action])
            if next_state == self.end_state:
                episode += 1
                self.mdp.reset()
                curr_state = self.start_state
            else:
                curr_state = next_state
            timestep+=1
                
        
    def run(self):
        np.random.seed(self.seed)
        if self.algorithm == 'sarsa':
            self.sarsa()
        elif self.algorithm == 'expected_sarsa':
            self.expected_sarsa()
        elif self.algorithm == 'q_learning':
            self.q_learning()
    