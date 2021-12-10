import RLlib as RL
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import shutil
# plot parameters

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

plt.rcParams.update({'font.size': 14})
episodes = 5
T = 80
critic = RL.Agents.RBFCritic()
actor = RL.Agents.OptimalControl(critic)
x0Max = 0.1
offline_training = False
breakFlag = True
disturber = RL.Agents.LPFRandomActor()
trace_P = False


def WriteSettings(testName):
    handle = open("Models/{0}/Settings.txt".format(testName), "w")
    description = "\n" +\
        "episodes = {0}\n".format(episodes) +\
        "T = {0}\n".format(T) +\
        "critic = {0}\n".format(type(critic)) +\
        "actor = {0}\n".format(type(actor)) +\
        "x0Max = {0}\n".format(x0Max) +\
        "offline_training = {0}\n".format(offline_training) +\
        "breakFlag = {0}\n".format(breakFlag) +\
        "disturber = {0}\n".format(type(disturber)) +\
        "trace_P = {0}\n".format(trace_P) +\
        str(RL.Constants())
    handle.write("Name: {0}\n".format(testName) + description)
    handle.close()


def reward(self, x, u, w):
    R = self.get('R')
    gamma_d = self.get('gamma_d')

    if self.breakCounter is not None:
        if self.breakCounter > 0:
            return -2 - R*u**2 + gamma_d**2*w**2
            #Glöm inte att ändra kriteriet i Trainer om du byter mellan
            #Rbf och linear!

    return torch.cos(x[0]) - 1 - R*u**2 + gamma_d**2*w**2


# L = RL.Scripts.trainAgents(critic, actor,  reward, x0Max, episodes, T,
#                            offline_training, breakFlag, disturber, trace_P)

# RL.Graphics.generatePlots(critic, actor, x0, L, T, disturber, testName)
# RL.Graphics.plot_simulation(actor, [3., 0], 10, disturber)
RL.Graphics.doAnimation(actor, [3., 0], 10, disturber, 'animation.gif')
# RL.Graphics.plot_heatmap(critic)
# RL.Graphics.plot_value_surface(critic)
plt.show()
