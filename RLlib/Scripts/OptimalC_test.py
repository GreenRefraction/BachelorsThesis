import RLlib
import matplotlib.pyplot as plt
from ..Environment import Environment
from ..Graphics import plot_value_surface
import numpy as np
from copy import deepcopy


def main():
    def Reward(self, x, u):
        R = self.get('R')
        return -x[0]**2 - R*u**2

    critic = RLlib.Agents.LinearCritic()
    actor = RLlib.Agents.OptimalControl(critic)
    x0 = [0.1, 0.]
    T = 20

    critic.get_true_solution()
    print(critic.P)
    plot_value_surface(critic)
    plt.title('Value function given by a riccati equation')

    fig = plt.figure()
    ENV = Environment(actor)
    (time, X, W, U) = ENV.Simulate(x0, T)
    plt.subplot(211)
    plt.title('Optimal control on a derived Value function, R: '+str(ENV.get('R')))
    plt.plot(time, X)
    plt.subplot(212)
    plt.plot(time, U)
    #plt.show()

    critic = RLlib.Agents.LinearCritic()
    actor = RLlib.Agents.OptimalControl(critic)
    Noise = RLlib.NoiseClass()
    trainer = RLlib.Trainer(critic, actor, Reward, Noise.LPFNNoise, False)
    Plist = []
    for ep in range(50):
        x0 = [np.random.uniform(-0.5, 0.5), 0]
        trainer.offline_train(x0, T)
        P = deepcopy(critic.P)
        P.requires_grad = False
        Plist.append(P.numpy())
        print(ep)

    # run a simulation with a trained critic
    ENV = Environment(actor)
    x0 = [0.1, 0]
    (time, X, W, U) = ENV.Simulate(x0, T)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('Optimal control of a trained linear critic')
    plt.plot(time, X)
    plt.subplot(212)
    plt.plot(time, U)

    Plist = np.array(Plist)
    fig = plt.figure()
    plt.plot(Plist[:, 0, 0], 'b', label='P00')
    plt.plot(Plist[:, 1, 0], 'r', label='P10')
    plt.plot(Plist[:, 1, 1], 'g', label='P11')
    critic.get_true_solution()
    P = critic.P.detach().numpy()
    print(Plist.shape)
    N = Plist.shape[0]
    plt.plot([0, N-1], [P[0, 0], P[0, 0]], '--b')
    plt.plot([0, N-1], [P[1, 0], P[1, 0]], '--r')
    plt.plot([0, N-1], [P[1, 1], P[1, 1]], '--g')
    plt.show()
