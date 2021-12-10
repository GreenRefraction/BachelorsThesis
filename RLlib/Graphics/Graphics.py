import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.animation import PillowWriter
import RLlib

plt.rcParams.update({
    "font.family":
    "serif",  # use serif/main font for text elements
    "text.usetex":
    True,  # use inline math for ticks
    "pgf.rcfonts":
    False,  # don't setup fonts from rc parameters
    "pgf.preamble": [
        "\\usepackage{units}",  # load additional packages
        "\\usepackage{metalogo}",
        "\\usepackage{unicode-math}",  # unicode math setup
        r"\setmathfont{xits-math.otf}",
        r"\setmainfont{DejaVu Serif}",  # serif font via preamble
    ]
})


def plot_value_surface(Critic, X=None, Xdot=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    M = 200
    N = 200
    Xmesh = np.linspace(-3, 3, M)
    Xdotmesh = np.linspace(-9, 9, N)

    Z = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x = torch.tensor([Xmesh[i], Xdotmesh[j]])
            Z[i, j] = float(Critic(x))

    Xmesh, Xdotmesh = np.meshgrid(Xmesh, Xdotmesh)

    Z = np.transpose(Z)
    surf = ax.plot_surface(Xmesh, Xdotmesh, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, alpha=0.5)
    if X is not None and Xdot is not None:
        Z = []
        for i, (x, xdot) in enumerate(zip(X, Xdot)):
            if abs(x) > 3 or abs(xdot) > 9:
                X = X[:i]
                Xdot = Xdot[:i]
                break
            input = torch.tensor([x, xdot])
            Z.append(Critic(input))
        plt.plot(X, Xdot, Z)

    plt.colorbar(surf)
    plt.xlabel('X')
    plt.ylabel('Xdot')


def doAnimation(actor, x0, T, disturber=None, savepath=None):
    env = RLlib.Environment(actor, disturber)
    (time, X, Xdot, U, W) = env.Simulate(x0, T)
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(121)
    ax2 = plt.subplot(222)
    ax3 = plt.subplot(224)

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    line1 = Line2D([], [], color='black')
    ax1.add_line(line1)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1, 1)
    ax1.set_aspect('equal', 'datalim')

    ax2.set_xlabel('t')
    ax2.set_ylabel('U')
    line2 = Line2D([], [], color='black')
    ax2.add_line(line2)
    ax2.set_xlim(0, T)
    umax = max(np.abs(U))
    ax2.set_ylim(-umax*1.2, umax*1.2)

    ax3.set_xlabel('t')
    ax3.set_ylabel('w')
    line3 = Line2D([], [], color='black')
    ax3.add_line(line3)
    ax3.set_xlim(0, T)
    wmax = max(W)
    wmin = min(W)
    ax3.set_ylim(wmin*1.2, wmax*1.2)

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        return line1, line2, line3

    def animate(frame, X, U, W, time):

        x = X[frame]
        x_data = [0, np.sin(x)]
        y_data = [0, np.cos(x)]
        t = time[:frame]
        u = U[:frame]
        w = W[:frame]

        line1.set_data(x_data, y_data)
        line2.set_data(t, u)
        line3.set_data(t, w)
        return line1, line2, line3

    anim = FuncAnimation(fig, animate, init_func=init, fargs=(X, U, W, time),
                         frames=len(X), interval=20, blit=True, repeat=True)
    if savepath is not None:
        writer = PillowWriter(fps=1/RLlib.Constants().get('dt'))
        anim.save(savepath, writer=writer)


def plot_simulation(actor, x0, T, disturber=None):
    # run a simulation with a trained critic
    ENV = RLlib.Environment(actor, disturber)
    (time, X, Xdot, U, W) = ENV.Simulate(x0, T)
    if disturber is not None:
        plt.figure(figsize=(8, 5))
        plt.subplot(311)
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelbottom=False) # labels along the bottom edge are off
        plt.plot(time, np.array(X)/np.pi)
        plt.ylabel(r'$\theta$  $[rad/\pi]$')
        plt.subplot(312)
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelbottom=False) # labels along the bottom edge are off
        plt.plot(time, U)
        plt.ylabel('u')
        plt.subplot(313)
        plt.plot(time, W)
        plt.grid(True)
        plt.ylabel('w')
        plt.xlabel('time [s]')
    else:
        plt.figure(figsize=(8, 4))
        plt.subplot(211)
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelbottom=False) # labels along the bottom edge are off
        plt.plot(time, np.array(X)/np.pi)
        plt.ylabel(r'$\theta$  $[rad/\pi]$')
        plt.subplot(212)
        plt.grid(True)
        plt.tick_params(
            axis='x',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            labelbottom=False) # labels along the bottom edge are off
        plt.plot(time, U)
        plt.ylabel('u')
        plt.xlabel('time [s]')


def plot_heatmap(critic, X=None, Xdot=None):
    plt.figure(figsize=(8, 5))
    M = 200
    N = 200
    Xmesh = np.linspace(-np.pi, np.pi, M)
    Xdotmesh = np.linspace(-9, 9, N)

    Z = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            x = torch.tensor([Xmesh[i], Xdotmesh[j]])
            Z[i, N-1-j] = float(critic(x))

    Xmesh, Xdotmesh = np.meshgrid(Xmesh, Xdotmesh)
    Z = np.transpose(Z)
    ax = plt.imshow(Z, extent=[-np.pi, np.pi, -9, 9], aspect='auto')
    if X is not None and Xdot is not None:
        Z = []
        for i, (x, xdot) in enumerate(zip(X, Xdot)):
            if abs(x) > 3 or abs(xdot) > 9:
                X = X[:i]
                Xdot = Xdot[:i]
                break
            input = torch.tensor([x, xdot])
            Z.append(critic(input))
        plt.plot(X, Xdot, Z)

    plt.colorbar(ax)
    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\omega$')
    tick_pos = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    tick_labels = [r'$-\pi$', r'$-\pi/2$', '0', r'$\pi/2$', r'$\pi$']
    plt.xticks(tick_pos, tick_labels)


def plot_loss(L):
    plt.figure()
    plt.semilogy(L)
    plt.title('Loss over episodes')
    plt.grid(True)
    plt.xlabel('Episode')
    plt.ylabel(r'$log(E)$')
