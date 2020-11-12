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

    def move(self,pos,dx,dy):
        x,y = pos
        if dx < 0:
            if dy > 0:
                return (max(0,x+dx),min(self.ncols-1,y+dy))
            else:
                return (max(0,x+dx),max(0,y+dy))
        else:
            if dy > 0:
                return (min(self.nrows-1,x+dx),min(self.ncols-1,y+dy))
            else:
                return (min(self.nrows-1,x+dx),max(0,y+dy))

    def step(self,action):
        wind_change = self.winds[self.curr_pos[1]]
        if self.stochastic and wind_change:
            wind_change = np.random.choice([wind_change,wind_change-1,wind_change+1])
        self.curr_pos = self.move(self.curr_pos,self.dx[action]-wind_change,self.dy[action])
        self.curr_state = self.pos_to_state(self.curr_pos)
        if self.curr_pos == self.end_pos:
            return(self.curr_state,0)
        else:
            return(self.curr_state,-1)

    def reset(self):
        self.curr_state = self.start_state
        self.curr_pos = self.start_pos
    




        

