import numpy as np
import os
import sys
import argparse
import random

class WindyGridworld(object):
    def __init__(self,nrows,ncols,start,end,winds,eight_moves=False,stochastic=False):
        self.nrows = nrows
        self.ncols = ncols
        self.start_pos = start
        self.end_pos = end
        self.winds = winds
        self.eight_moves = eight_moves
        self.stochastic = stochastic
        self.start_state = self.pos_to_state(self.start_pos)
        self.end_state = self.pos_to_state(self.end_pos)
        self.num_states = self.nrows*self.ncols
        self.curr_pos = self.start_pos
        self.curr_state = self.start_state
        if not self.eight_moves:
            #Actions: 0->Up 1->Right 2->Down- 3->Left
            self.dx = [-1,0,1,0]
            self.dy = [0,1,0,-1]
        
        else:
            #Actions: 0->Up 1->Right 2->Down 3->Left 4->Up-Right 5->Down-Right 6->Down-Left 7->Up-Left
            self.dx = [-1,0,1,0,-1,1,1,-1]
            self.dy = [0,1,0,-1,1,1,-1,-1]

    def pos_to_state(self,pos):
        return pos[0]*self.ncols + pos[1]

    def check_pos(self,pos):
        x,y = pos
        if x < 0:
            x = 0
        elif x >= self.nrows:
            x = self.nrows-1
        if y < 0:
            y = 0
        elif y >= self.ncols:
            y = self.ncols-1
        
        return (x,y)

    def step(self,action):
        wind_change = self.winds[self.curr_pos[1]]
        x_new = self.curr_pos[0] + self.dx[action] - wind_change
        y_new = self.curr_pos[1] + self.dy[action]
        if self.stochastic and wind_change:
            next_pos = (random.choice([x_new+1,x_new,x_new-1]),y_new)
        else:
            next_pos = (x_new,y_new)
        next_pos = self.check_pos(next_pos)
        self.curr_pos = next_pos
        self.curr_state = self.pos_to_state(self.curr_pos)
        print(action)
        print(next_pos)
        if self.curr_pos == self.end_pos:
            return(self.curr_state,0)
        else:
            return(self.curr_state,-1)

    def reset(self):
        self.curr_state = self.start_state
        self.curr_pos = self.start_pos
    




        

