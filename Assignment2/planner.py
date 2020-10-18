from pulp import pulp,LpProblem,LpVariable,value,LpMaximize
import os
import argparse
import numpy as np

class MDP(object):
    def __init__(self,num_states=None,num_actions=None,transition_matrix=None,reward_matrix=None,discount=None,start_state=None,end_states=None,mdptype=None):
        """Initialize the MDP object
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.discount = discount
        self.start_state = start_state
        self.end_states = end_states
        self.mdptype = mdptype
    
    def parse(self,mdpfile):
        """Parse the given MDP file

        Args:
            mdpfile (string): Path to mdp file
        """
        with open(mdpfile,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line == '':
                    continue
                elif line.startswith("numStates"):
                    self.num_states = int(line.split(" ")[-1].strip())

                elif line.startswith("numActions"):
                    self.num_actions = int(line.split(" ")[-1].strip())

                elif line.startswith("start"):
                    self.start_state = int(line.split(" ")[-1].strip())

                elif line.startswith("end"):
                    self.end_states = [int(x.strip()) for x in line.split(" ")[1:] if x != ""]
                    if self.end_states[0] == -1:
                        self.end_states =  []
                    self.transition_matrix = np.zeros((self.num_states,self.num_actions,self.num_states))
                    self.reward_matrix = np.zeros((self.num_states, self.num_actions, self.num_states))

                elif line.startswith("transition"):
                    five_tuple = [x.strip() for x in line.split(" ")[1:] if x!= ""]
                    s1,ac,s2,r,p = int(five_tuple[0]), int(five_tuple[1]), int(five_tuple[2]), float(five_tuple[3]), float(five_tuple[4])
                    self.transition_matrix[s1,ac,s2] = p
                    self.reward_matrix[s1,ac,s2] = r
                
                elif line.startswith("mdptype"):
                    self.mdptype = line.split(" ")[-1].strip()

                elif line.startswith("discount"):
                    self.discount = float(line.split(" ")[-1].strip())

        self.optimum_value = np.zeros(self.num_states)
        self.optimum_policy = np.zeros(self.num_states,dtype=int)

    def max_norm(self,a,b):
        """Max norm of difference of 2 vectors

        Args:
            a (numpy.ndarray): vector 1
            b (numpy.ndarray): vector2
        """
        return np.max(np.abs(a-b))

    def value_from_policy(self,policy):
        """Get the value function for the given  policy

        Args:
            policy (numpy.ndarray): Policy for each state
        """
        value = np.zeros(self.num_states)
        non_terminal_states = np.delete(np.arange(self.num_states),self.end_states)
        new_index = np.arange(self.num_states)*self.num_actions + policy
        new_t_matrix = self.transition_matrix.reshape(-1,self.num_states)[new_index] #num_states * num_states
        new_r_matrix = self.reward_matrix.reshape(-1,self.num_states)[new_index] #num_states * num_states
        t_r = np.sum(new_t_matrix*new_r_matrix,axis=-1) #num_states
        new_t_matrix = new_t_matrix[non_terminal_states,:][:,non_terminal_states] #num_nt_states * num_nt_states
        new_r_matrix = new_r_matrix[non_terminal_states,:][:,non_terminal_states] #num_nt_states * num_nt_states
        t_r = t_r[non_terminal_states] #num_nt_states

        non_terminal_values = np.linalg.inv(np.eye(len(non_terminal_states))-self.discount*new_t_matrix)@t_r
        value[non_terminal_states] = non_terminal_values
        return value



    def policy_from_value(self,value):
        """Get the **optimum** policy function for the  values

        Args:
            policy (numpy.ndarray): Policy for each state
        """
        return np.argmax(np.sum(self.transition_matrix*(self.reward_matrix + self.discount*value.reshape((1,1,-1))),axis=-1),axis=-1)

    def value_iteration(self):
        """Get the optimum values and policy using value iteration
        """
        old_value = self.optimum_policy
        while(True):
            new_value = np.max(np.sum(self.transition_matrix*(self.reward_matrix + self.discount*old_value.reshape((1,1,-1))),axis=-1),axis=-1)
            if self.max_norm(new_value,old_value) < 1e-8:
                break
            old_value = new_value

        self.optimum_value = new_value
        self.optimum_policy = self.policy_from_value(self.optimum_value)
        self.optimum_value = self.value_from_policy(self.optimum_policy)

    def linear_programming(self):
        """Get the optimum values and policy using linear programming formulation
        """
        state_variables = []
        for i in range(self.num_states):
            state_variables.append(LpVariable("s"+str(i)))
        state_variables = np.array(state_variables)
        prob = LpProblem("optimumValue",LpMaximize)
        prob += -np.sum(state_variables)
        for i in range(self.num_states):
            for j in range(self.num_actions):
                prob += state_variables[i] >= np.sum(self.transition_matrix[i,j,:]*(self.reward_matrix[i,j,:]+self.discount*state_variables))
        _ = pulp.PULP_CBC_CMD(msg=0).solve(prob)
        optimum_value = []
        for i in range(self.num_states):
            optimum_value.append(value(state_variables[i]))
        self.optimum_value = np.array(optimum_value)
        self.optimum_policy = self.policy_from_value(self.optimum_value)
        self.optimum_value = self.value_from_policy(self.optimum_policy)

    def howard_policy_iteration(self):
        """Get the optimum values and policy using howard policy iteration
        """
        old_policy = self.optimum_policy
        while(True):
            values = self.value_from_policy(old_policy)
            new_policy = self.policy_from_value(values)
            if (old_policy==new_policy).all():
                break
            old_policy = new_policy

        self.optimum_policy = new_policy
        self.optimum_value = self.value_from_policy(self.optimum_policy)

    def print_optimum(self):
        """Print out optimum policy and value for each state
        """
        for i in range(self.num_states):
            print('{0:.6f}'.format(self.optimum_value[i]),self.optimum_policy[i])

    def print_mdp(self):
        """Print the MDP in the same format as mdpfile
        """
        print("numStates",self.num_states)
        print("numActions",self.num_actions)
        print("start",self.start_state)
        if len(self.end_states) == 0:
            es_string = "-1"
        else:
            es_string = " ".join([str(x) for x in self.end_states])
        print("end",es_string)
        ts_string = ""
        first = 0
        for i in range(self.num_states):
            for j in range(self.num_actions):
                for k in range(self.num_states):
                    if self.transition_matrix[i,j,k] != 0:
                        cells = ["transition",i,j,k,self.reward_matrix[i,j,k],self.transition_matrix[i,j,k]]
                        if not first:
                            first = 1
                        else:
                            ts_string +="\n"
                        ts_string += " ".join(str(x) for x in cells)
        print(ts_string)
        print("mdptype",self.mdptype)
        print("discount",self.discount)

    def solve_from_file(self,planfile):
        with open(planfile,'r') as f:
            lines = f.readlines()
        for i,line in enumerate(lines):
            line = line.strip()
            if line == "":
                continue
            v = float(line.split(" ")[0].strip())
            p = int(line.split(" ")[-1].strip())
            self.optimum_value[i] = v
            self.optimum_policy[i] = p



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mdp',type=str,help='Path to mdp file')
    parser.add_argument('--algorithm',type=str,help='Algorithm for MDP planning',choices=['vi','hpi','lp'])
    args = parser.parse_args()
    mdp = MDP()
    mdp.parse(args.mdp)
    if args.algorithm == 'vi':
        mdp.value_iteration()
    elif args.algorithm == 'hpi':
        mdp.howard_policy_iteration()
    elif args.algorithm == 'lp':
        mdp.linear_programming()
    mdp.print_optimum()
