from agent import Agent
from windy_gridworld import WindyGridworld
import matplotlib.pyplot as plt
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows',type=int,default=7,help='Number of Rows')
    parser.add_argument('--ncols',type=int,default=10,help='Number of Columns')
    parser.add_argument('--start',type=int,nargs=2,default=(3,0),help='Start coordinates')
    parser.add_argument('--end',type=int,nargs=2,default=(3,7),help='End coordinates')
    parser.add_argument('--winds',type=int, nargs='+',default=[0,0,0,1,1,1,2,2,1,0], help='Wind values')
    parser.add_argument('--timesteps',type=int,default=10000,help='Number of timesteps')
    parser.add_argument('--algorithms',type=str,nargs='+',default=['sarsa'],help='Algorithms for control')
    parser.add_argument('--stochastic',action='store_true',help='Whether the wind is stochastic in nature')
    parser.add_argument('--eight_moves',action='store_true',help='Whether eight moves are allows')
    parser.add_argument('--colors',type=str,nargs='+',help='Colors to label the plot')
    parser.add_argument('--title',type=str,help='Title of the plot')
    parser.add_argument('--seeds',type=str,default=10,help='Number of independent seed runs for the experiment')
    parser.add_argument('--epsilon',type=float,default=0.1,help='Epsilon for epsilon-greedy algorithm')
    parser.add_argument('--lr',type=float,default=0.5,help='Learning Rate')
    args = parser.parse_args()
    
    mdp = WindyGridworld(args.nrows,args.ncols,tuple(args.start),tuple(args.end),args.winds,args.eight_moves,args.stochastic)
    plt.figure()
    plt.grid()
    plt.title(args.title)
    plt.xlabel('Timesteps')
    plt.ylabel('Episodes')
    algorithms_map = {'SARSA':'sarsa','Expected SARSA':'expected_sarsa','Q Learning':'q_learning'}
    for i,algorithm in enumerate(args.algorithms):
        episodes_array = []
        for seed in range(args.seeds):
            agent = Agent(algorithms_map[algorithm],mdp,seed,args.timesteps,args.epsilon,args.lr)
            agent.run()
            episodes_array.append(agent.episodes)
        episodes_array = np.mean(np.array(episodes_array),axis=0)
        plt.plot(np.arange(args.timesteps),episodes_array,label=algorithm,color=args.colors[i])
    plt.legend()
    plt.savefig(args.title+'.png')