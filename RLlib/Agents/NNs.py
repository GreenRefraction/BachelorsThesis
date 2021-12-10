import torch
import torch.nn as nn
import numpy as np
import RLlib.Functions as fcn
import scipy.linalg
from RLlib.Params import Constants
from . import torch_rbf as rbf


def Gauss(x):
    return torch.exp(-x*x)


class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(2, 15)
        self.fc2 = nn.Linear(15, 8)
        self.fc3 = nn.Linear(8, 1)

    def forward(self, x):
        x2 = torch.tensor([x[0]**2, x[1]**2], requires_grad=True)
        x = torch.cat((x, x2), 0)
        x = self.fc1(x)
        return x


class Actor(nn.Module):

    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(2, 15)
        self.fc2 = nn.Linear(15, 15)
        self.fc3 = nn.Linear(15, 1)

    def forward(self, x):
        x = Gauss(self.fc1(x))
        x = Gauss(self.fc2(x))
        x = self.fc3(x)
        return x


class TestCritic(nn.Module):
    """Small network for testing etrace algorithm"""

    def __init__(self):
        super(TestCritic, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = Gauss(self.fc1(x))
        x = self.fc2(x)
        return x


class TestActor(nn.Module):
    """Small network for testing etrace algorithm"""

    def __init__(self):
        super(TestActor, self).__init__()
        self.fc1 = nn.Linear(2, 2)
        self.fc2 = nn.Linear(2, 1)

    def forward(self, x):
        x = Gauss(self.fc1(x))
        x = self.fc2(x)
        return x


class RandomActor(nn.Module):
    def __init__(self):
        super(RandomActor, self).__init__()
        self.N = 100
        self.noiseList = [0]*self.N

    def forward(self, x):
        self.noiseList.append(np.random.normal(0, 1.5))
        if len(self.noiseList) > self.N:
            del self.noiseList[0]
        return torch.tensor(np.mean(self.noiseList))


class LPFRandomActor(nn.Module, Constants):
    def __init__(self):
        nn.Module.__init__(self)
        Constants.__init__(self)
        self.dt = self.get('dt')
        self.tau_n = self.get('tau_n')
        self.u = 0

    def forward(self, x):
        dt = self.dt
        tau_n = self.tau_n
        self.u = self.u * (1 - dt/tau_n) + dt/tau_n * np.random.normal(0, 0.7)
        return torch.tensor(self.u)/self.get('wMax')


class OptimalControl(nn.Module, Constants):
    def __init__(self, Critic):
        nn.Module.__init__(self)
        Constants.__init__(self)
        self.R = self.get('R')
        self.m = self.get('m')
        self.L = self.get('L')
        self.Critic = Critic

    def forward(self, x):
        R = self.get('R')
        x_track = x.clone().detach()
        x_track.requires_grad = True
        dVdx = torch.autograd.grad(self.Critic(x_track), x_track)[0]
        # dVdx = -2*torch.matmul(self.Critic.P.float(), x)
        return 0.5/R*torch.dot(
                        torch.tensor([0, 1/(self.m*self.L**2)]),
                        dVdx)


class OptimalDisturber(nn.Module, Constants):
    def __init__(self, Critic):
        nn.Module.__init__(self)
        Constants.__init__(self)
        self.Critic = Critic

    def forward(self, x):
        gamma = self.get('gamma_d')
        x_track = x.clone().detach()
        x_track.requires_grad = True
        dVdx = torch.autograd.grad(self.Critic(x_track), x_track)[0]
        g1 = torch.tensor([0., 1.])
        return -1/(2*gamma**2)*torch.dot(g1, dVdx)


class RBFNetwork(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        N = 15
        self.rbf1 = rbf.RBF(2, N*N, fcn.gaussian)
        theta = np.linspace(-np.pi, np.pi, N)
        omega = np.linspace(-5/2*np.pi, 5/2*np.pi, N)
        center = [[x, w] for x in theta
                  for w in omega]
        self.rbf1.centres = torch.tensor(center, requires_grad=False)
        # self.rbf1.sigmas = nn.Parameter(0.2*torch.ones(N*N), True)
        self.lin1 = nn.Linear(N*N, 1)
        # self.lin1.weight = nn.Parameter(torch.ones(1, N*N), True)

    def forward(self, x):
        out = self.rbf1(x)
        out = self.lin1(out)
        return out


# The following has been copied from:
# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
class RBFCritic(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        N = 15
        self.rbf1 = rbf.RBF(2, N*N, fcn.gaussian)
        theta = np.linspace(-np.pi, np.pi, N)
        omega = np.linspace(-5/2*np.pi, 5/2*np.pi, N)
        center = [[x, w] for x in theta
                  for w in omega]
        self.rbf1.centres = torch.tensor(center, requires_grad=False)
        # self.rbf1.sigmas = nn.Parameter(0.2*torch.ones(N*N), True)
        self.lin1 = nn.Linear(N*N, 1)
        # self.lin1.weight = nn.Parameter(torch.ones(1, N*N), True)

    def forward(self, x):
        while abs(x[0]) > np.pi:
            if x[0] > 0:
                x[0] -= 2*np.pi
            else:
                x[0] += 2*np.pi
        out = self.rbf1(x)
        out = self.lin1(out)
        return out


class RBFActor(nn.Module, Constants):
    def __init__(self):
        nn.Module.__init__(self)
        Constants.__init__(self)
        N = 30
        self.rbf1 = rbf.RBF(2, N*N, fcn.gaussian)
        theta = np.linspace(0, 2*np.pi, N)
        omega = np.linspace(-5/2*np.pi, 5/2*np.pi, N)
        center = [[x, w] for x in theta
                  for w in omega]
        self.rbf1.centres = torch.tensor(center, requires_grad=False)
        # self.rbf1.sigmas = nn.Parameter(0.2*torch.ones(N*N), True)
        self.lin1 = nn.Linear(N*N, 1)
        # self.lin1.weight = nn.Parameter(torch.ones(1, N*N), True)

    def forward(self, x):
        x[0] = x[0] % (2*np.pi)
        out = self.rbf1(x)
        out = self.lin1(out)
        self.get('uMax') * torch.tanh(out)
        return out


class LinearCritic(nn.Module, Constants):
    def __init__(self):
        nn.Module.__init__(self)
        Constants.__init__(self)
        self.P = torch.ones((2, 2), requires_grad=True) #replace: rand -> ones
        self.double()

    def get_true_solution(self):
        g = self.get('g')
        L = self.get('L')
        m = self.get('m')
        my = self.get('my')
        tau = self.get('tau')
        A = np.array([[0, 1], [g/L, -my/(m*L*L)]])
        B2 = np.array([0, 1/(m*L*L)])
        B1 = np.array([0, 1])
        B = np.array([B1, B2]).T
        Q = np.array([[1, 0], [0, 0]])

        R = self.get('R')
        gamma = self.get('gamma_d')
        R = np.array([[-gamma**2, 0], [0, R]])

        P = scipy.linalg.solve_continuous_are(A-np.identity(2)/(2*tau), B, Q, R)
        self.P = torch.tensor(P, requires_grad=True)*10

    def parameters(self):
        return [self.P]

    def forward(self, x):
        return - self.P[0, 0]*x[0]**2\
               - self.P[0, 1]*x[1]*x[0]\
               - self.P[1, 0]*x[1]*x[0]\
               - self.P[1, 1]*x[1]**2
