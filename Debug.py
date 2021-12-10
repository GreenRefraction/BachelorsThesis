import matplotlib.pyplot as plt
import numpy as np
import RLlib as RL
import torch


critic = torch.load("Models/Success5/model.pt")
actor = RL.Agents.OptimalControl(critic)
disturber = RL.Agents.LPFRandomActor()

ENV = RL.Environment(actor, disturber)
x0 = [3., 0.]
T = 20
N = 200
time = None
x = np.zeros(1000)
x2 = np.zeros(1000)
u = np.zeros(1000)
u2 = np.zeros(1000)
w = np.zeros(1000)
w2 = np.zeros(1000)
for i in range(N):
    (time, X, Xdot, U, W) = ENV.Simulate(x0, T)
    print(i)
    x += X / N
    x2 += X**2 / N
    u += U / N
    u2 += U**2 / N
    w += W / N
    w2 += W**2 / N
Sx = np.sqrt(x2 - x**2)
Su = np.sqrt(u2 - u**2)
Sw = np.sqrt(w2 - w**2)
if disturber is not None:
    plt.figure(figsize=(8, 5))
    plt.subplot(311)
    plt.grid(True)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        labelbottom=False) # labels along the bottom edge are off
    plt.plot(time, x/np.pi, 'black')
    plt.plot(time, (x + Sx)/np.pi, '--g')
    plt.plot(time, (x - Sx)/np.pi, '--g')
    plt.ylabel(r'$<\theta> \pm \sigma_\theta$  $[rad/\pi]$')
    plt.subplot(312)
    plt.grid(True)
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        labelbottom=False) # labels along the bottom edge are off
    plt.plot(time, u, 'black')
    plt.plot(time, u + Su, '--g')
    plt.plot(time, u - Su, '--g')
    plt.ylabel(r'$<u> \pm \sigma_u$')
    plt.subplot(313)
    plt.plot(time, w, 'black')
    plt.plot(time, w + Sw, '--g')
    plt.plot(time, w - Sw, '--g')
    plt.grid(True)
    plt.ylabel(r'$<w> \pm \sigma_x$')
    plt.xlabel('time [s]')
plt.show()
