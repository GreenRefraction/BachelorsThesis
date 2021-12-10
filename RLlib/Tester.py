import RLlib as RL
import torch
import numpy as np


class Tester(RL.Trainer):
    def __init__(self, Critic, Actor, Reward, Noise, breakFlag = False, printEtrace = False):
        self.printEtrace = printEtrace #boolean
        super().__init__(Critic, Actor, Reward, Noise, breakFlag)

    def updateCritic(self, x, V, delta, etrace):
        """Overwrite Trainers updateCritic function with manual calc"""
        dt = self.dt
        #Manually calculating dVdw:
        paramList = []
        for param in self.Critic.parameters():
            paramList.append(param)
        W1 = paramList[0].clone().detach() #matrix
        b1 = paramList[1].clone().detach() #vector
        W2 = paramList[2].clone().detach() #vector
        b2 = paramList[3].clone().detach() #scalar
        x_untrack = x.clone().detach()
        z1 = torch.matmul(W1, x_untrack) + b1
        dadz1 = torch.tensor([ [-2 * z1[0] * np.exp(-z1[0]**2), 0], [0, -2 * z1[1] * np.exp(-z1[1]**2)] ])
        dz1dW1 = torch.tensor([ [x[0], x[1], 0 ,0], [0, 0, x[0], x[1]] ])

        dVdW1 = torch.matmul(torch.matmul(W2, dadz1)[0], dz1dW1)
        dVdW1 = torch.reshape(dVdW1, (2,2))
        dVdb1 = torch.matmul(W2,dadz1)
        dVdW2 = torch.tensor([np.exp(-z1[0]**2), np.exp(-z1[1]**2)])
        dVdb2 = torch.tensor([1])
        dVdwList = [dVdW1, dVdb1, dVdW2, dVdb2]

        if self.printEtrace == True:
            print("etrace :" + str(etrace) + "\n")

        i = 0
        # etrace is not updated if zip is used!
        for param in self.Critic.parameters():
            dim = param.shape
            dVdw = torch.reshape(dVdwList[i], dim)
            etrace[i] = (1 - dt/self.kappa) * etrace[i] + dt * dVdw
            param.grad = -delta * etrace[i]
            param.data.sub_(self.CriticLR * param.grad)
            i += 1

        self.Coptimizer.zero_grad()
        return etrace

    """
    def TD_error(self, x, x_, u):
        #Overwrite Trainers TD_error function with manual calc
        assert x[0] != x_[0]
        assert x[1] != x_[1]
        dt = self.dt
        tau = self.tau
        V = self.Critic(x)
        # For manually calculating dVdx:
    s    paramList = []
        for param in self.Critic.parameters():
            paramList.append(param)
        W1 = paramList[0].clone().detach()  # matrix
        b1 = paramList[1].clone().detach()  # vector
        W2 = paramList[2].clone().detach()  # vector
        b2 = paramList[3].clone().detach()  # scalar
        x_untrack = x.clone().detach()
        z1 = torch.matmul(W1,x_untrack) + b1
        dadz1 = torch.tensor([ [-2 * z1[0] * np.exp(-z1[0]**2), 0], [0, -2 * z1[1] * np.exp(-z1[1]**2)] ])
        dVdx = torch.matmul(torch.matmul(W2, dadz1), W1)[0]

        dVdt = torch.dot(dVdx, (x - x_)/dt)
        if self.breakCounter is not None and self.breakCounter >= 1:
            r = -1
        else:
            r = self.Reward(self, x, u)
        delta = r - V / tau + dVdt

        return 0.5 * delta * delta, delta
    """
