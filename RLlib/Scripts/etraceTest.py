from math import cos
import time as pyTime
import copy
import numpy as np
import matplotlib.pyplot as plt
import torch
import RLlib as RL
import RLlib.Agents as Agents #import MAPS like this. Not modules.



def main():
    """Set optimizer to SGD. Enable seed. """
    testCritic = Agents.TestCritic()
    testCritic2 = copy.deepcopy(testCritic) #copy
    testActor = Agents.TestActor()
    testActor2 = copy.deepcopy(testActor)
    # a noise generating function:
    LPFNNoise = RL.NoiseClass().LPFNNoise #make sure that seed is enabled in params!
    def reward(self, x, u):
        return cos(x[0])
    breakFlag = True
    actorTrainable = True
    #trainer object:
    trainer = RL.Trainer(testCritic, testActor, reward, LPFNNoise, breakFlag)
    T = 20
    dt = 0.02
    x0 = [np.pi, 0.]
    episodes = 5

    #Value V and signal U before and after training for trainer object
    V0trainer = trainer.Critic(torch.tensor(x0)).detach().numpy()
    U0trainer = trainer.Actor(torch.tensor(x0)).detach().numpy()
    print('trainer \n')
    for ep in range(episodes):
        print("ep "+str(ep))
        (time, X, W, U, E) = trainer.online_train(x0, T, return_U = True, return_E=True) #train one episode
    Vtrainer = trainer.Critic(torch.tensor(x0)).detach().numpy()
    Utrainer = trainer.Actor(torch.tensor(x0)).detach().numpy()

    LPFNNoise2 = RL.NoiseClass().LPFNNoise #Will generate same noise as LPFNNoise because of seed.
    #tester object
    breakFlag = True
    tester = RL.Tester(testCritic2, testActor2, reward, LPFNNoise2, breakFlag) #Tester
 
    #Value V and signal U before and after training for tester object
    V0tester = tester.Critic(torch.tensor(x0)).detach().numpy()
    U0tester = tester.Actor(torch.tensor(x0)).detach().numpy()
    print('\n tester \n')
    for ep in range(episodes):
        print("ep "+str(ep))
        print(tester.breakCounter)
        (time2, X2, W2, U2, E2) = tester.online_train(x0, T, return_U = True, return_E=True) #train one episode
    Vtester = tester.Critic(torch.tensor(x0)).detach().numpy()
    Utester = tester.Actor(torch.tensor(x0)).detach().numpy()

    digits = 4

    print(V0trainer)
    print(V0tester)
    print(U0trainer)
    print(U0tester)
    print(Vtrainer)
    print(Vtester)
    print(Utrainer)
    print(Utester)

    assert V0trainer.round(digits) == V0tester.round(digits)
    assert U0trainer.round(digits) == U0tester.round(digits)
    assert Vtrainer.round(digits) == Vtester.round(digits)
    assert Utrainer.round(digits) == Utester.round(digits)
    print("Tests passed! \n")

    pyTime.sleep(5)
    
    #Another test:
    T = 0.1
    dt = 0.02
    x0 = [np.pi, 0.]
    episodes = 3

    testCritic3 = Agents.TestCritic()
    testActor3 = Agents.TestActor()
    LPFNNoise3 = RL.NoiseClass().LPFNNoise
    printEtrace = True
    tester3 = RL.Tester(testCritic3, testActor3, reward, LPFNNoise3, breakFlag, printEtrace)

    print("Printing etrace to make sure it resets after each episode: \n")
    for ep in range(episodes):
        print("ep "+str(ep) + "\n")
        (time3, X3, W3, U3, E3) = tester3.online_train(x0, T, return_U = True, return_E=True) #train one episode
