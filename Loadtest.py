import RLlib as RL
import numpy as np
import matplotlib.pyplot as plt
import torch

testName = "Testkladd"
simName = "TestkladdSim"

#critic = torch.load("Models\Success4_v4\model.pt")
critic = torch.load("Models/{0}/model.pt".format(testName))

actor = RL.Agents.OptimalControl(critic)
disturber = RL.Agents.OptimalDisturber(critic)

handle = open("Models/{0}/Loss.txt".format(testName), "r")
L = handle.read()
handle.close()

L = L[1:-1]

L = L.split(",")
lossList = []
for loss in L:
    lossList.append(float(loss))


T = 20
x0 = [0.1, 0.]

RL.Graphics.generatePlots(critic, actor, x0, lossList, T, disturber, simName)
plt.show()