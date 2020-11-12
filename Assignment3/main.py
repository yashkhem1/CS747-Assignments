from agent import Agent
from windy_gridworld import WindyGridworld
import matplotlib.pyplot as plt

if __name__ == "__main__":
    mdp = WindyGridworld(7,10,(4,0),(4,7),[0,0,0,1,1,1,2,2,1,0])
    agent_sarsa = Agent('q_learning',mdp,10,100,0.1,0.5)
    plt.figure()
    agent_sarsa.run()
    agent_sarsa.plot_graphs(plt,'r')
    plt.savefig('a.png')