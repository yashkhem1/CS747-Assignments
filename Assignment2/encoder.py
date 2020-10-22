import argparse
import numpy as np
from planner import MDP
import time

class Maze(MDP):
    def __init__(self):
        """Initialize the Maze object
        """
        super(Maze,self).__init__()
        self.rows = None
        self.columns = None
        self.matrix = None
        self.state_index_map = None
        self.index_state_map = None
        
    def parse(self,gridfile,partial=False):
        """Parse the Maze object from the grid file

        Args:
            gridfile (str): Path to grid file
            partial (bool, optional): Whether to calculate transition matrix and reward matrix. Defaults to False.
        """
        with open(gridfile,'r') as f:
            grid_mat = []
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            cells = [int(x.strip()) for x in line.split(" ") if x!=""]
            grid_mat.append(cells)
        
        self.matrix = np.array(grid_mat)
        self.rows,self.columns = self.matrix.shape
        self.num_actions = 4 #{0:"North",1:"East",2:"South",3:"West"}
        self.state_index_map = np.argwhere(self.matrix!=1)
        self.num_states = len(self.state_index_map)
        self.index_state_map = np.cumsum(self.matrix!=1).reshape(self.rows,self.columns)-1
        self.index_state_map[self.matrix==1] = -1
        self.start_state = self.index_state_map[self.matrix==2][0]
        self.end_states = self.index_state_map[self.matrix==3]
        self.discount = 0.99
        self.mdptype = 'episodic'
        self.optimum_value = np.zeros(self.num_states)
        self.optimum_policy = np.zeros(self.num_states)
        
        if not partial:
            self.transition_matrix = np.zeros((self.num_states,self.num_actions,self.num_states))
            self.reward_matrix = np.zeros((self.num_states,self.num_actions,self.num_states))
            # x_coord, y_coord = np.where(self.matrix!=1)
            # states = np.tile(np.arange(self.num_states),4)
            # bool_not_end = np.tile(self.matrix[x_coord,y_coord]!=3,4)
            # actions = np.arange(4).repeat(self.num_states)
            # transition_states = np.concatenate([self.index_state_map[x_coord-1,y_coord],self.index_state_map[x_coord,y_coord+1],self.index_state_map[x_coord+1,y_coord],self.index_state_map[x_coord,y_coord-1]])
            # bool_miss_wall = np.concatenate([self.matrix[x_coord-1,y_coord]!=1,self.matrix[x_coord,y_coord+1]!=1,self.matrix[x_coord+1,y_coord]!=1,self.matrix[x_coord,y_coord-1]!=1])
            # bool_reach_end = np.concatenate([self.matrix[x_coord-1,y_coord]==3,self.matrix[x_coord,y_coord+1]==3,self.matrix[x_coord+1,y_coord]==3,self.matrix[x_coord,y_coord-1]==3])
            # b1 = bool_miss_wall*bool_not_end
            # b2 = ~bool_miss_wall*bool_not_end
            # b3 = bool_miss_wall*bool_not_end*bool_reach_end
            # b4 = bool_miss_wall*bool_not_end*~bool_reach_end
            # self.transition_matrix[states[b1],actions[b1],transition_states[b1]] = 1
            # self.transition_matrix[states[b2],actions[b2],states[b2]] = 1
            # self.reward_matrix[states[b3],actions[b3],transition_states[b3]] = 10000
            # self.reward_matrix[states[b4],actions[b4],transition_states[b4]] = -10000/self.num_states
            # self.reward_matrix[states[b2],actions[b2],states[b2]] = -10000

            for i in range(self.num_states):
                if i in self.end_states:
                    continue
                x,y = self.state_index_map[i]
                min_factor = max(1e-10,self.discount**(2*(self.rows+self.columns)))
                max_factor = 1/min_factor

                #West
                if y>0 and self.matrix[x,y-1]!=1:
                    self.transition_matrix[i,3,self.index_state_map[x,y-1]] = 1
                    if self.index_state_map[x,y-1] in self.end_states:
                        self.reward_matrix[i,3,self.index_state_map[x,y-1]] = max_factor*1e5
                    else:
                        self.reward_matrix[i,3,self.index_state_map[x,y-1]] = -1
                else:
                    self.transition_matrix[i,3,i] = 1
                    self.reward_matrix[i,3,i] = -1e3

                #East
                if y<self.columns-1 and self.matrix[x,y+1]!=1:
                    self.transition_matrix[i,1,self.index_state_map[x,y+1]] = 1
                    if self.index_state_map[x,y+1] in self.end_states:
                        self.reward_matrix[i,1,self.index_state_map[x,y+1]] = max_factor*1e5
                    else:
                        self.reward_matrix[i,1,self.index_state_map[x,y+1]] = -1
                else:
                    self.transition_matrix[i,1,i] = 1
                    self.reward_matrix[i,1,i] = -1e3

                #North
                if x>0 and self.matrix[x-1,y]!=1:
                    self.transition_matrix[i,0,self.index_state_map[x-1,y]] = 1
                    if self.index_state_map[x-1,y] in self.end_states:
                        self.reward_matrix[i,0,self.index_state_map[x-1,y]] = max_factor*1e5
                    else:
                        self.reward_matrix[i,0,self.index_state_map[x-1,y]] = -1
                else:
                    self.transition_matrix[i,0,i] = 1
                    self.reward_matrix[i,0,i] = -1e3

                #South
                if x<self.rows-1 and self.matrix[x+1,y]!=1:
                    self.transition_matrix[i,2,self.index_state_map[x+1,y]] = 1
                    if self.index_state_map[x+1,y] in self.end_states:
                        self.reward_matrix[i,2,self.index_state_map[x+1,y]] = max_factor*1e5
                    else:
                        self.reward_matrix[i,2,self.index_state_map[x+1,y]] = -1
                else:
                    self.transition_matrix[i,2,i] = 1
                    self.reward_matrix[i,2,i] = -1e3
    
    def get_path(self):
        """Generate path from the optimal policy and value

        Returns:
            List(str): List of directions to follow from the start state
        """
        directions = ['N','E','S','W']
        dx = [-1,0,1,0]
        dy = [0,1,0,-1]
        path = []
        curr_state = self.start_state
        while(curr_state not in self.end_states):
            curr_x, curr_y = self.state_index_map[curr_state]
            action = int(self.optimum_policy[curr_state])
            path.append(directions[action])
            curr_state = self.index_state_map[curr_x+dx[action],curr_y+dy[action]]
        return path

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid',type=str,help='Path to grid file')
    args = parser.parse_args()
    maze = Maze()
    maze.parse(args.grid)
    maze.print_mdp()

    

                



            

    
        




