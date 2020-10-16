from bandit import MultiArmedBandit
from algorithms import epsilon_greedy,ucb,kl_ucb,thompson_sampling,thompson_sampling_with_hint
import numpy as np
import os
import sys
import pickle
import matplotlib.pyplot as plt

def generateOutput1():
    instances = ['../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt']
    algorithms = [epsilon_greedy,ucb,kl_ucb,thompson_sampling]
    algo_strings = ['epsilon-greedy','ucb','kl-ucb','thompson-sampling']
    seeds = np.arange(50)
    horizons = [100,400,1600,6400,25600,102400]
    w = open('outputDataT1.txt','w')
    plot_dict = {}
    for instance in instances:
        plot_dict[instance] = {}
        for algorithm,algo_string in zip(algorithms,algo_strings):
            total_regrets = np.zeros(len(horizons))
            for seed in seeds:
                np.random.seed(seed)
                p_list = []
                with open(instance,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            p_list.append(float(line.replace("\n","").strip()))
                        except:
                            continue
                mab = MultiArmedBandit(p_list)
                if algorithm == epsilon_greedy:
                    regrets = epsilon_greedy(mab,0.02,horizons)
                elif algorithm == kl_ucb:
                    regrets = kl_ucb(mab,0,1e-3,horizons)
                else:
                    regrets = algorithm(mab,horizons)
                
                for horizon in regrets.keys():
                    out_string = instance+", "+algo_string+", "+str(seed)+", 0.02, "+str(horizon)+", "+str(regrets[horizon])
                    w.write(out_string+"\n")
                    print(out_string)
                total_regrets += np.array(list(regrets.values()))
            plot_dict[instance][algo_string] = total_regrets/len(seeds)
    

    with open('outputDataT1.pkl','wb') as p:
        pickle.dump(plot_dict,p)


def generateOutput2():
    instances = ['../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt']
    algorithms = [thompson_sampling, thompson_sampling_with_hint]
    algo_strings = ['thompson-sampling','thompson-sampling-with-hint']
    seeds = np.arange(50)
    horizons = [100,400,1600,6400,25600,102400]
    w = open('outputDataT2.txt','w')
    plot_dict = {}
    for instance in instances:
        plot_dict[instance] = {}
        for algorithm,algo_string in zip(algorithms,algo_strings):
            total_regrets = np.zeros(len(horizons))
            for seed in seeds:
                np.random.seed(seed)
                p_list = []
                with open(instance,'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        try:
                            p_list.append(float(line.replace("\n","").strip()))
                        except:
                            continue
                mab = MultiArmedBandit(p_list)
                regrets = algorithm(mab,horizons)
                
                for horizon in regrets.keys():
                    out_string = instance+", "+algo_string+", "+str(seed)+", 0.02, "+str(horizon)+", "+str(regrets[horizon])
                    w.write(out_string+"\n")
                    print(out_string)
                total_regrets += np.array(list(regrets.values()))
            plot_dict[instance][algo_string] = total_regrets/len(seeds)
    

    with open('outputDataT2.pkl','wb') as p:
        pickle.dump(plot_dict,p)


def plot_data():
    horizons = [100,400,1600,6400,25600,102400]

    with open('outputDataT1.pkl','rb') as f:
        regretsT1 = pickle.load(f)
    files = ['i1T1.pdf','i2T1.pdf','i3T1.pdf']
    keys = ['../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt']
    algorithms = ['epsilon-greedy','ucb','kl-ucb','thompson-sampling']
    colors = ['r','g','b','y']
    for (key,file1) in zip(keys,files):
        regrets = regretsT1[key]
        plt.figure()
        plt.xscale('log')
        for (algo,color) in zip(algorithms,colors):
            plt.plot(horizons,regrets[algo],color=color,label=algo)
        plt.legend()
        plt.xlabel('Horizon')
        plt.ylabel('Regret')
        plt.savefig(file1)

    with open('outputDataT2.pkl','rb') as f:
        regretsT2 = pickle.load(f)
    files = ['i1T2.pdf','i2T2.pdf','i3T2.pdf']
    keys = ['../instances/i-1.txt','../instances/i-2.txt','../instances/i-3.txt']
    algorithms = ['thompson-sampling','thompson-sampling-with-hint']
    colors = ['r','g']
    for (key,file2) in zip(keys,files):
        regrets = regretsT2[key]
        plt.figure()
        plt.xscale('log')
        for (algo,color) in zip(algorithms,colors):
            plt.plot(horizons,regrets[algo],color=color,label=algo)
        plt.legend()
        plt.xlabel('Horizon')
        plt.ylabel('Regret')
        plt.savefig(file2)


if __name__ == "__main__":
    # generateOutput1()
    # generateOutput2()
    plot_data()

                
