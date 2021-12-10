import RLlib
import matplotlib.pyplot as plt
import torch
import os
import shutil
import math


def generatePlots(critic, actor, x0, L, T, disturber=None, testName="default"):
    """Plots loss, theta-t, u-t, valuefunc landscape, does animation
    for trained critic and actor.
    Input: critic, actor, loss list, simulation time T"""
    path = "Models/{0}".format(testName)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    path += "/"
    torch.save(critic, path + "model.pt")

    RLlib.Graphics.plot_loss(L)
    plt.savefig(path + 'LossPlot.png')
    handle = open("Models/{0}/Loss.txt".format(testName), "w")
    handle.write(str(L))
    handle.close()

    # run a simulation with a trained critic

    (time, X, Xdot, U, W) = RLlib.Graphics.plot_simulation(actor, x0, T, disturber)
    plt.savefig(path + "Simulation.png")

    RLlib.Graphics.plot_value_surface(critic, X=X, Xdot=Xdot)
    plt.savefig(path + "ValueLandscape.png")

    RLlib.Graphics.plot_heatmap(critic)
    plt.savefig(path + "Heatmap.png")

    # RLlib.Graphics.doAnimation(X, U, W, time, savepath= None)#path+"animation.gif")
