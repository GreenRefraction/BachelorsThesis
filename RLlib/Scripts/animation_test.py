import RLlib
import matplotlib.pyplot as plt

import RLlib.Agents.torch_rbf as rbf
import time as pyTime
from ..Graphics import doAnimation
import numpy as np

def main():
    #critic = Agents.LinearCritic()
    #actor = Agents.OptimalControl(critic)
    layer_widths = [2, 1]
    layer_centres = [15,15]
    basis_func = rbf.gaussian
    RBFCritic = RLlib.Agents.RBFCritic(layer_widths, layer_centres, basis_func)
    RBFActor = RLlib.Agents.RBFActor(layer_widths, layer_centres, basis_func)
    def reward(self, x, u):
        return -x[0]**2 - u**2


    critic = RLlib.Agents.LinearCritic()
    actor = RLlib.Agents.Actor()

    trainer = RLlib.Trainer(critic, actor, reward)
    T = 20
    dt = 0.02
    episodes = 101 #100
    runs = 1

    ErrorMatrix = []
    AVG = np.zeros(episodes)
    RMSE = np.zeros(episodes)

    try:
        for run in range(runs):
            print('Run: '+str(run))
            ErrorList = []
            for ep in range(episodes):
                #x0 = [np.random.uniform(-np.pi, np.pi), 0]
                x0 = [np.pi, 0.]
                (time, X, W, t_up, U, E) = trainer.train(x0, T, return_U = True, return_E=True)
                Emean = np.mean(E)
                ErrorList.append(Emean)
                trainer.sigma0 = 0.5 / (1 + ep**0.5)
                print(ep, t_up, Emean)
                if ep % 20 == 0:
                    doAnimation(X, U, time)
                    #plt.show(block = True)

            #plt.plot([i for i in range(episodes)], ErrorList,'*')
            ErrorMatrix.append(ErrorList)
            AVG += ErrorList
        AVG /= runs

        for run in range(runs):
            RMSE += (ErrorMatrix[run] - AVG)**2
        RMSE /= runs

    except KeyboardInterrupt:
        pass
