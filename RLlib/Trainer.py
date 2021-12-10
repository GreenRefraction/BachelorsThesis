from .Params import Constants
import numpy as np
import torch
from .Environment import Environment
import torch.optim as optim
import RLlib.Agents as Agents
from copy import deepcopy


class Trainer(Constants):
    def __init__(self, Critic, Actor, Reward, Noise, breakFlag=False, Disturber = None):
        """Critic: trainable NN object, input x, return V.
        Actor: NN object, Input x, return u.
        Reward: function, input self, x, u, return r
        Noise: function, input noise, return noise
        breakflag: boolean"""
        super(Trainer, self).__init__()
        self.Critic = Critic
        self.Actor = Actor
        self.Reward = Reward
        self.Noise = Noise
        if not breakFlag:
            self.breakCounter = None
        else:
            self.breakCounter = 0
        self.CriticLR = self.get('C_lr')
        self.ActorLR = self.get('A_lr')
        self.optimizer = self.get('optimizer')
        trainableActors = [Agents.Actor,
                           Agents.RBFActor,
                           Agents.TestActor,
                           Agents.Neural_Network]
        self.actorTrainable = type(Actor) in trainableActors

        if self.optimizer == "SGD":
            self.Coptimizer = optim.SGD(Critic.parameters(), lr=self.CriticLR)
            if self.actorTrainable:
                self.Aoptimizer = optim.SGD(Actor.parameters(), lr=self.ActorLR)
        elif self.optimizer == "Adam":
            self.Coptimizer = optim.Adam(Critic.parameters(), lr=self.CriticLR)
            if self.actorTrainable:
                self.Aoptimizer = optim.Adam(Actor.parameters(), lr=self.ActorLR)
        self.dt = self.get('dt')
        self.sigma0 = self.get('sigma0')
        self.tau = self.get('tau')
        self.kappa = self.get('kappa')
        self.noise0 = self.get('noise0')
        self.uMax = self.get('uMax')
        self.wMax = self.get('wMax')
        etrace0 = []
        for param in self.Critic.parameters(): #myParameters, parameters()
            dim = param.shape
            etraceElem = torch.zeros(dim)
            etrace0.append(etraceElem)
        self.etrace0 = etrace0
        self.Disturber = Disturber

    def online_train(self, x0, T, return_U=False, return_E=False, return_W=False):
        """Train both the actor and critic over time T, initial conditions x0
        timesteps N. Training is done online."""
        dt = self.dt
        N = round(T/dt)
        x_ = torch.tensor(x0, requires_grad=False)
        time = np.linspace(0, T, N)
        sigma = self.sigma0
        etrace = self.etrace0.copy()
        (X, Xdot, U, E, W) = ([], [], [], [], [])
        ENV = Environment(self.Actor)
        u_, A_, noise = self.compute_control_law(x_, self.noise0)
        u_ = u_.detach()
        w_ = self.compute_disturbance(x_)
        for t in time:
            x = ENV.step(x_, u_, w_)
            u, A, noise = self.compute_control_law(x, noise)
            w = self.compute_disturbance(x)
            loss = self.online_update(x, x_, u, A, etrace, noise, sigma, w)
            self.appendData(u, x, loss, w, U, X, Xdot, E, W)
            # Check for overrotation
            if self.breakCheck(x_, t):
                break

            x_ = x
            u_ = u
            A_ = A
            w = w_

        returnVal = (time, np.array(X), np.array(Xdot))
        if return_U:
            returnVal += (np.array(U),)
        if return_E:
            returnVal += (np.array(E),)
        if return_W:
            returnVal += (np.array(W),)
        return returnVal

    def offline_train(self, x0, T):
        """Train both the actor and critic over time T, initial conditions x0
        timesteps N. Training is done offline"""
        dt = self.dt
        N = round(T/dt)
        x_ = torch.tensor(x0, requires_grad=False)
        time = np.linspace(0, T, N)
        ENV = Environment(self.Actor)
        R = []
        X = []
        noiseList = []
        u_, A_, noise = self.compute_control_law(x_, self.noise0)
        w_ = self.compute_disturbance(x_)
        for t in time:
            x = ENV.step(x_, u_, w_)
            u, A, noise = self.compute_control_law(x, noise)
            w = self.compute_disturbance(x)
            # Check for overrotation
            if self.breakCheck(x_, t):
                time = time[time < t]
                break
            R.append(float(self.Reward(self, x, u, w)))
            X.append(x.clone())
            noiseList.append(noise)
            x_ = x
            u_ = u
            w_ = w
        R = np.array(R)
        return self.offline_update(time, X, R, noiseList)

    def online_update(self, x, x_, u, A, etrace, noise, sigma, w):
        """Calls updateCritic, updateActor, TD_error.
        Returns loss. Called in online_traning."""
        loss, delta = self.TD_error(x, x_, u, w)
        V = self.Critic(x)
        self.updateCritic(x, V, delta, etrace)

        if self.actorTrainable:
            self.updateActor(A, delta, noise, sigma)
        return loss

    def offline_update(self, time, X, R, noiseList):
        L = []
        # --- Reversed from T to 0, should be linear
        rev_time = range(len(time)-1)[::-1]
        G = 0.
        decay = np.exp(-self.dt/self.tau)
        # etrace = self.etrace0
        for t in rev_time:
            G *= decay
            # add the trapezoid integration between step t and t+1
            G += (R[t] + decay * R[t + 1]) / 2 * self.dt
            delta = G - self.Critic(X[t])
            loss = delta ** 2
            # self.updateCritic(X[t], self.Critic(X[t]), delta, etrace)
            L.append(float(loss))
            loss.backward()
            self.Coptimizer.step()
            self.Coptimizer.zero_grad()
            if self.actorTrainable:
                A = self.Actor(X[t])
                self.updateActor(A, delta, noiseList[t], 0)
        # ----The old one ----
        # for t, x, noise in zip(time, X, noiseList):
        #    s = time[time > t]
        #    decay = np.exp((t-s)/self.tau)
        #    G = np.trapz(decay*R[time > t], x=s)
        #    delta = G - self.Critic(x)
        #    loss = delta**2
        #    L.append(float(loss))
        #    loss.backward()
        #    self.Coptimizer.step()
        #    self.Coptimizer.zero_grad()
        #    if self.actorTrainable:
        #        A = self.Actor(x)
        #        self.updateActor(A, delta, noise, 0)

        return L

    def compute_control_law(self, x, noise):
        """Returns control signal u, actor output A and noise. The control
        signal has noise added to it if actorTrainable = True. Called by
        online_training and offline_training."""
        noise = self.Noise(noise)
        if self.actorTrainable:
            noiseTerm = self.sigma0 * noise
        else:
            noiseTerm = 0
        A = self.Actor(x)
        u = self.uMax * 2/np.pi*torch.atan(np.pi/2 * (A + noiseTerm))
        u = u.detach()
        return u, A, noise

    def compute_disturbance(self, x):
        if self.Disturber is not None:
            #D = self.Disturber(x)
            #w = self.wMax * 2/np.pi*torch.atan(np.pi/2*D)
            w = self.Disturber(x)
            w = self.get('wMax') * 2/np.pi*torch.atan(np.pi/2 * w)
        else:
            w = 0
        return w

    def updateCritic(self, x, V, delta, etrace):
        """Updates gradients using etrace. Does one optimizer step for Critic.
        Resets gradient. Called by online_update. """
        # loss = delta**2
        # loss.backward()
        V.backward()
        i = 0
        dt = self.dt
        # etrace is not updated when zip is used!
        for param in self.Critic.parameters():
            etrace[i] = (1 - dt/self.kappa) * etrace[i] + dt*param.grad #mult dt
            param.grad = -delta * etrace[i] 
            i += 1
        self.Coptimizer.step()
        self.Coptimizer.zero_grad()
        return etrace

    def updateActor(self, A, delta, noise, sigma):
        """Does one optimizer step for Actor.
        Resets gradient. Called by online_update. """
        A.backward()
        for param in self.Actor.parameters():
            param.grad *= -delta * noise
        self.Aoptimizer.step()
        self.Aoptimizer.zero_grad()

    def TD_error(self, x, x_, u, w):
        """Calculates TD_error. Called by online_update."""
        dt = self.dt
        V = self.Critic(x) #Only filter here and the line below?
        V_ = self.Critic(x_) #filter
        r = self.Reward(self, x, u, w)
        gamma = self.get('gamma')
        delta = r + (gamma * V - V_)/dt
        return 0.5 * delta * delta, delta

    def breakCheck(self, x_, t):
        """Returns break command 1 second after overrotation,
        if break functionality is enabled."""
        dt = self.dt
        # Check if break functionality is enabled:
        if self.breakCounter is not None:
            # Keep increasing break counter:
            if self.breakCounter >= 1:
                self.breakCounter += 1
            # Break after 1 second:
            if self.breakCounter*dt >= 1.:
                print("breaking at t = "+str(t))
                self.breakCounter = 0  # reset counter
                return True
        # Check for overrotation and if break functionality is enabled:
        if abs(float(x_[0])) > 5*np.pi and self.breakCounter == 0:
            self.breakCounter += 1

        return False

    def appendData(self, u, x, loss, w, U, X, Xdot, E, W):
        """Appends data calculated in online_update"""
        U.append(float(u))
        X.append(float(x[0]))
        Xdot.append(float(x[1]))
        E.append(float(loss))
        W.append(float(w))
