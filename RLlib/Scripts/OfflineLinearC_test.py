import RLlib as RL
import matplotlib.pyplot as plt
import time as pyTime
from ..Graphics import *
import numpy as np


def main():
    def Reward(self, x, u):
        R = self.get('R')
        return -x[0]**2 - R * u**2

    critic = RL.Agents.LinearCritic()
    actor = RL.Agents.OptimalControl(critic)

    trainer = RL.Trainer(critic, actor, Reward)
    T = 20
    for ep in range(5):
        x0 = [np.random.uniform(-0.1, 0.1), 0.]
        trainer.offline_train(x0, T)
        print(ep)
    plot_value_surface(critic)
    plt.show()
    ENV = RL.Environment(actor)
    x0 = [0.1, 0]
    (time, X, W, U) = ENV.Simulate(x0, T)
    plt.subplot(211)
    plt.plot(time, X)
    plt.subplot(212)
    plt.plot(time, U)
    plt.show()
