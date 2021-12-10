import torch
from math import sin
import numpy as np
from .Params import Constants


class Environment(Constants):
    def __init__(self, Actor, Disturber=None):
        Constants.__init__(self)
        self.Actor = Actor
        self.Disturber = Disturber
        # parameters of the system
        self.L = self.get('L')
        self.g = self.get('g')
        self.m = self.get('m')
        self.my = self.get('my')
        self.dt = self.get('dt')
        self.uMax = self.get('uMax')
        self.wMax = self.get('wMax')

    def step(self, x, u, w):
        """Updates x with one timestep. This is a RK4 step"""

        dt = self.dt
        k1 = self.f(x, u, w) * dt
        k2 = self.f(x + k1/2, u, w)*dt
        k3 = self.f(x + k2/2, u, w)*dt
        k4 = self.f(x + k3, u, w)*dt
        return x + (k1 + 2*k2 + 2*k3 + k4)/6

    def f(self, x, u, w):
        """Returns a vector containing angular velocity and angular
         acceleration  at time t, in state x with the control law u"""
        g = self.g
        L = self.L
        my = self.my
        m = self.m
        return torch.tensor([x[1],
                             g/L*sin(x[0])-my/(m*L**2)*x[1] + u/(m*L**2) + w])

    def Simulate(self, x0, T):
        """Simulate the system with the current iteration of the actor.
        This should be used without exploration and with trained NNs"""
        dt = self.dt
        N = round(T/dt)
        time = np.linspace(0, T, N)
        (X, Xdot, U, W) = ([], [], [], [])
        x = torch.tensor(x0)
        for t in time:
            A = self.Actor(x)
            u = self.uMax * 2/np.pi * torch.atan(A * np.pi/2)
            if self.Disturber is not None:
                #D = self.Disturber(x)
                #w = self.wMax * 2/np.pi*torch.atan(np.pi/2*D)
                w = self.Disturber(x)
                w = self.get('wMax') * 2/np.pi * torch.atan(w * np.pi/2)
            else:
                w = 0
            x = self.step(x, u, w)
            U.append(float(u))
            X.append(float(x[0]))
            Xdot.append(float(x[1]))
            W.append(float(w))
        return (time, np.array(X), Xdot, np.array(U), np.array(W))
