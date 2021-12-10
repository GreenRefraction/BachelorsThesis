import RLlib as RL
import numpy as np
import matplotlib.pyplot as plt
import torch


def WriteSettings(testName, episodes):
    handle = open("Models/{0}/Settings.txt".format(testName), "w")
    description = "Linear Critic, {0} episodes, optimal ".format(episodes)+\
    "control. Online training, 20 seconds, "+\
    "reward = -x^2 -Ru^2 + gamma^2*w^2, -0.2**2 - R*uMax**2 + gamma_d**2*w**2"+\
    ", breakFlag = True, disturber = None, x0Max = 0.1., dt*dV/dw, "+\
    "No w filtering."+\
    str(RL.Constants())
    handle.write("Name: {0}\n".format(testName) + description)
    handle.close()

def plotP(Plist):
    criticExact = RL.Agents.LinearCritic()
    criticExact.get_true_solution() 
    Pexact = criticExact.parameters()[0]
    pLen = len(Plist[:, 0, 0])
    P00e = float(Pexact[0, 0]) * np.ones((pLen))
    P10e = float(Pexact[1, 0]) * np.ones((pLen))
    P11e = float(Pexact[1, 1]) * np.ones((pLen))
    plt.figure()
    plt.plot(Plist[:, 0, 0], label='P00')
    plt.plot(Plist[:, 1, 0], label='P10')
    plt.plot(Plist[:, 1, 1], label='P11')
    plt.plot(P00e, label = 'P00_exact')
    plt.plot(P10e, label = 'P10_exact')
    plt.plot(P11e, label = 'P11_exact') 
    plt.legend()
    path = "Models/{0}".format(testName) + "/"
    plt.savefig(path + "pValues.png")
    plt.show()
    
def reward(self, x, u, w):
    R = self.get('R')
    gamma_d = self.get('gamma_d')
    uMax = self.get('uMax')
    
    if self.breakCounter is not None:
        if self.breakCounter > 0:
            return -0.2**2 - R*uMax**2 + gamma_d**2*w**2
    return -x[0]**2 - R*u**2 + gamma_d**2*w**2 
 
episodes = 1000
T = 20
testName = "Test1"

critic = RL.Agents.LinearCritic()
#critic = torch.load("model.pt")
actor = RL.Agents.OptimalControl(critic) 
x0Max = 0.1
offline_training = False
breakFlag = True
disturber = None #RL.Agents.OptimalDisturber(critic)
trace_P = True
L, Plist = RL.Scripts.trainAgents(critic, actor,  reward, x0Max, episodes, T,
                       offline_training, breakFlag, disturber, trace_P)
plt.close('all')
RL.Graphics.generatePlots(critic, actor, L, T, disturber, testName)
plotP(Plist)

WriteSettings(testName, episodes)


