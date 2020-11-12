import numpy as np
import os
import sys
import random
import argparse
from windy_gridworld import WindyGridworld

class Agent(object):
    def __init__(self,algorithm,mdp:WindyGridworld,nseeds,num_timestamps,epsilon,lr):
        self.algorithm = algorithm
        self.mdp = mdp
        self.seeds = range(nseeds)
        self.num_timestamps = num_timestamps
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
        self.seeds = range(nseeds)
        self.timesteps_array = []
        self.episodes_array = []
    

    def sarsa(self):
        timesteps = []
        episodes = []
        timestep = 0
        episode = 0
        curr_state = self.start_state
        if random.random() < self.epsilon:
            action = random.randint(0,self.num_actions-1)
        else:
            action = np.argmax(self.q_matrix[curr_state])
        while timestep < self.num_timestamps:
            timesteps.append(timestep)
            episodes.append(episode)
            next_state,reward = self.mdp.step(action)
            if next_state == self.end_state:
                self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(reward - self.q_matrix[curr_state,action])
                episode += 1
                print(episode)
                self.mdp.reset()
                curr_state = self.start_state
                if random.random() < self.epsilon:
                    action = random.randint(0,self.num_actions-1)
                else:
                    action = np.argmax(self.q_matrix[curr_state])
            else:
                if random.random() < self.epsilon:
                    next_action = random.randint(0,self.num_actions-1)
                else:
                    next_action = np.argmax(self.q_matrix[next_state])
                self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(reward + self.q_matrix[next_state,next_action] - self.q_matrix[curr_state,action])
                curr_state = next_state
                action = next_action

            timestep+=1
        self.timesteps_array.append(timesteps)
        self.episodes_array.append(episodes)

    def expected_sarsa(self):
        timesteps = []
        episodes = []
        timestep = 0
        episode = 0
        curr_state = self.start_state
        while timestep < self.num_timestamps:
            timesteps.append(timestep)
            episodes.append(episode)
            if random.random() < self.epsilon:
                action = random.randint(0,self.num_actions-1)
            else:
                action = np.argmax(self.q_matrix[curr_state])
            next_state,reward = self.mdp.step(action)
            expectation = (1-self.epsilon)*np.max(self.q_matrix[next_state]) + self.epsilon*np.mean(self.q_matrix[next_state])
            target = reward + expectation
            self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(target - self.q_matrix[curr_state,action])
            if next_state == self.end_state:
                episode += 1
                print(episode)
                self.mdp.reset()
                curr_state = self.start_state
            else:
                curr_state = next_state
            timestep+=1
        self.timesteps_array.append(timesteps)
        self.episodes_array.append(episodes)
        

    def q_learning(self):
        timesteps = []
        episodes = []
        timestep = 0
        episode = 0
        curr_state = self.start_state
        while timestep < self.num_timestamps:
            timesteps.append(timestep)
            episodes.append(episode)
            if random.random() < self.epsilon:
                action = random.randint(0,self.num_actions-1)
            else:
                action = np.argmax(self.q_matrix[curr_state])
            next_state,reward = self.mdp.step(action)
            target = reward + np.max(self.q_matrix[next_state])
            self.q_matrix[curr_state,action] = self.q_matrix[curr_state,action] + self.lr*(target - self.q_matrix[curr_state,action])
            if next_state == self.end_state:
                episode += 1
                # print(episode)
                self.mdp.reset()
                curr_state = self.start_state
            else:
                curr_state = next_state
            timestep+=1
                
        self.timesteps_array.append(timesteps)
        self.episodes_array.append(episodes)

    def run(self):
        for seed in self.seeds:
            random.seed(seed)
            if self.algorithm == 'sarsa':
                self.sarsa()
            elif self.algorithm == 'expected_sarsa':
                self.expected_sarsa()
            elif self.algorithm == 'q_learning':
                self.q_learning()


    def plot_graphs(self,plt,color):
        mean_episodes = np.mean(np.array(self.episodes_array),axis=0)
        mean_timesteps = np.mean(np.array(self.timesteps_array),axis=0)
        plt.plot(mean_timesteps,mean_episodes,color)