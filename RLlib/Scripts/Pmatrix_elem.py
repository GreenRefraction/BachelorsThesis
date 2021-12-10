import RLlib
import matplotlib.pyplot as plt
import numpy as np
from copy import deepcopy


def reward(self, x, u, w):
    R = self.get('R')
    gamma = self.get('gamma_d')
    return -x[0]**2 - R*u**2 + gamma**2*w**2


def main():
    critic = RLlib.Agents.LinearCritic()
    actor = RLlib.Agents.OptimalControl(critic)
    disturber = None

    Noise = RLlib.NoiseClass().LPFNNoise
    trainer = RLlib.Trainer(critic, actor, reward, Noise)

    T = 20
    episodes = 500
    Plist = []
    L = []
    for ep in range(episodes):
        x0 = [np.random.uniform(-0.5, 0.5), 0]
        (_, _, _, loss) = trainer.online_train(x0, T, return_E=True)
        Plist.append(deepcopy(critic.P.detach().numpy()))
        print(ep)
    Plist = np.array(Plist)
    print(Plist.shape)
    plt.plot(Plist[:][0, 0], label='P00')
    plt.plot(Plist[:][1, 0], label='P10')
    plt.plot(Plist[:][1, 1], label='P11')
    plt.legend()

    RLlib.Graphics.plot_value_surface(critic)

    RLlib.Graphics.generatePlots(critic, actor, L, T)
    plt.show()
