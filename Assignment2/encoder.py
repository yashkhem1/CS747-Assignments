import argparse
import numpy as np
from planner import MDP

class Maze(MDP):
    def __init__(self):
        super(Maze,self).__init__()
        self.rows = None
        self.columns = None
        self.matrix = None
        self.state_index_map = None
        self.index_state_map = None
        
    def parse(self,gridfile,partial=False):
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
        self.discount = 0.9
        self.mdptype = 'episodic'
        self.optimum_value = np.zeros(self.num_states)
        self.optimum_policy = np.zeros(self.num_states)
        
        if not partial:
            self.transition_matrix = np.zeros((self.num_states,self.num_actions,self.num_states))
            self.reward_matrix = np.zeros((self.num_states,self.num_actions,self.num_states))
            x_coord, y_coord = np.where(self.matrix!=1)
            states = np.arange(self.num_states)
            bool_not_end = self.matrix[x_coord,y_coord]!=3
            #West
            transition_states = self.index_state_map[x_coord,y_coord-1]
            bool_miss_wall = self.matrix[x_coord,y_coord-1]!=1
            bool_reach_end = self.matrix[x_coord,y_coord-1]==3
            self.transition_matrix[:,3,:][states[bool_miss_wall*bool_not_end],transition_states[bool_miss_wall*bool_not_end]] = 1
            # self.transition_matrix[:,3,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = 1
            self.reward_matrix[:,3,:][states[bool_miss_wall*bool_not_end*bool_reach_end],transition_states[bool_miss_wall*bool_not_end*bool_reach_end]] = 10000
            self.reward_matrix[:,3,:][states[bool_miss_wall*bool_not_end*~bool_reach_end],transition_states[bool_miss_wall*bool_not_end*~bool_reach_end]] = -1
            # self.reward_matrix[:,3,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = -10000

            #East
            transition_states = self.index_state_map[x_coord,y_coord+1]
            bool_miss_wall = self.matrix[x_coord,y_coord+1]!=1
            bool_reach_end = self.matrix[x_coord,y_coord+1]==3
            self.transition_matrix[:,1,:][states[bool_miss_wall*bool_not_end],transition_states[bool_miss_wall*bool_not_end]] = 1
            # self.transition_matrix[:,1,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = 1
            self.reward_matrix[:,1,:][states[bool_miss_wall*bool_not_end*bool_reach_end],transition_states[bool_miss_wall*bool_not_end*bool_reach_end]] = 10000
            self.reward_matrix[:,1,:][states[bool_miss_wall*bool_not_end*~bool_reach_end],transition_states[bool_miss_wall*bool_not_end*~bool_reach_end]] = -1
            # self.reward_matrix[:,1,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = -10000

            #North
            transition_states = self.index_state_map[x_coord-1,y_coord]
            bool_miss_wall = self.matrix[x_coord-1,y_coord]!=1
            bool_reach_end = self.matrix[x_coord-1,y_coord]==3
            self.transition_matrix[:,0,:][states[bool_miss_wall*bool_not_end],transition_states[bool_miss_wall*bool_not_end]] = 1
            # self.transition_matrix[:,0,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = 1
            self.reward_matrix[:,0,:][states[bool_miss_wall*bool_not_end*bool_reach_end],transition_states[bool_miss_wall*bool_not_end*bool_reach_end]] = 10000
            self.reward_matrix[:,0,:][states[bool_miss_wall*bool_not_end*~bool_reach_end],transition_states[bool_miss_wall*bool_not_end*~bool_reach_end]] = -1
            # self.reward_matrix[:,0,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = -10000

            #South
            transition_states = self.index_state_map[x_coord+1,y_coord]
            bool_miss_wall = self.matrix[x_coord+1,y_coord]!=1
            bool_reach_end = self.matrix[x_coord+1,y_coord]==3
            self.transition_matrix[:,2,:][states[bool_miss_wall*bool_not_end],transition_states[bool_miss_wall*bool_not_end]] = 1
            # self.transition_matrix[:,2,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = 1
            self.reward_matrix[:,2,:][states[bool_miss_wall*bool_not_end*bool_reach_end],transition_states[bool_miss_wall*bool_not_end*bool_reach_end]] = 10000
            self.reward_matrix[:,2,:][states[bool_miss_wall*bool_not_end*~bool_reach_end],transition_states[bool_miss_wall*bool_not_end*~bool_reach_end]] = -1
            # self.reward_matrix[:,2,:][states[~bool_miss_wall*bool_not_end],states[~bool_miss_wall*bool_not_end]] = -10000
            # for i in range(self.num_states):
            #     if i in self.end_states:
            #         continue
            #     x,y = self.state_index_map[i]

            #     #West
            #     if y>0 and self.matrix[x,y-1]!=1:
            #         self.transition_matrix[i,3,self.index_state_map[x,y-1]] = 1
            #         if self.index_state_map[x,y-1] in self.end_states:
            #             self.reward_matrix[i,3,self.index_state_map[x,y-1]] = 70
            #         else:
            #             self.reward_matrix[i,3,self.index_state_map[x,y-1]] = -1
            #     else:
            #         self.transition_matrix[i,3,i] = 1
            #         self.reward_matrix[i,3,i] = -25

            #     #East
            #     if y<self.columns-1 and self.matrix[x,y+1]!=1:
            #         self.transition_matrix[i,1,self.index_state_map[x,y+1]] = 1
            #         if self.index_state_map[x,y+1] in self.end_states:
            #             self.reward_matrix[i,1,self.index_state_map[x,y+1]] = 70
            #         else:
            #             self.reward_matrix[i,1,self.index_state_map[x,y+1]] = -1
            #     else:
            #         self.transition_matrix[i,1,i] = 1
            #         self.reward_matrix[i,1,i] = -25

            #     #North
            #     if x>0 and self.matrix[x-1,y]!=1:
            #         self.transition_matrix[i,0,self.index_state_map[x-1,y]] = 1
            #         if self.index_state_map[x-1,y] in self.end_states:
            #             self.reward_matrix[i,0,self.index_state_map[x-1,y]] = 70
            #         else:
            #             self.reward_matrix[i,0,self.index_state_map[x-1,y]] = -1
            #     else:
            #         self.transition_matrix[i,0,i] = 1
            #         self.reward_matrix[i,0,i] = -25

            #     #South
            #     if x<self.rows-1 and self.matrix[x+1,y]!=1:
            #         self.transition_matrix[i,2,self.index_state_map[x+1,y]] = 1
            #         if self.index_state_map[x+1,y] in self.end_states:
            #             self.reward_matrix[i,2,self.index_state_map[x+1,y]] = 70
            #         else:
            #             self.reward_matrix[i,2,self.index_state_map[x+1,y]] = -1
            #     else:
            #         self.transition_matrix[i,2,i] = 1
            #         self.reward_matrix[i,2,i] = -25
    
    def get_path(self):
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

    

                



            

    
        




