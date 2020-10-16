from algorithms import kl_ucb,epsilon_greedy,ucb,thompson_sampling,thompson_sampling_with_hint
from bandit import MultiArmedBandit
import numpy as np

if __name__ == "__main__":
    # total_regrets = np.zeros(6)
    # for seed in range(50):
    #     print(seed)
    #     np.random.seed(seed)
    #     # mab = MultiArmedBandit([0.4,0.3,0.5,0.2,0.1])
    #     # mab = MultiArmedBandit([0.15,0.23,0.37,0.44,0.50,0.32,0.78,0.21,0.82,0.56,0.34,0.56,0.84,0.76,0.43,0.65,0.73,0.92,0.10,0.89,0.48,0.96,0.60,0.54,0.49])
    #     mab = MultiArmedBandit([0.4,0.8])
    #     regrets = thompson_sampling(mab,[100,400,1600,6400,25600,102400])
    #     total_regrets += np.array(list(regrets.values()))
    # print(total_regrets/50)

    ### T3 ###
    files = ['../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt']
    epsilons = [0.0002,0.02,0.9]
    for file_ in files:
        for epsilon in epsilons:
            total_regrets = np.zeros(6)
            for seed in range(50):
                np.random.seed(seed)
                p_list = []
                with open(file_,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            p_list.append(float(line.replace("\n","").strip()))
                        except:
                            continue
                mab = MultiArmedBandit(p_list)
                regrets = epsilon_greedy(mab,epsilon,[100,400,1600,6400,25600,102400])
                total_regrets += np.array(list(regrets.values()))
            print(file_,epsilon,total_regrets/50)