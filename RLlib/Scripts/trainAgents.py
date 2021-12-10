import RLlib
import numpy as np
from copy import deepcopy


def trainAgents(critic, actor,  reward, x0Max, episodes, T, offline_training=False, breakFlag=False, disturber=None, trace_P=False):
    """ """
    Noise = RLlib.NoiseClass()
    trainer = RLlib.Trainer(critic, actor, reward, Noise.LPFNNoise, breakFlag, disturber)
    L = []
    Plist = []
    for ep in range(episodes):
        x0 = [np.random.uniform(-x0Max, x0Max), 0]
        if offline_training:
            loss = trainer.offline_train(x0, T)
        else:
            (_, _, _, loss) = trainer.online_train(x0, T, return_E=True)
        if trace_P:
            Plist.append(deepcopy(critic.P.detach().numpy()))
        trainer.sigma0 = trainer.get('sigma0')/(1 + 99*ep/episodes)
        L.append(float(np.mean(loss)))
        meanLoss = '{:.2e}'.format(np.mean(loss))
        sigma0 = '{:.2e}'.format(trainer.sigma0)
        print("{0}/{1}\t| {2}\t{3}".format(ep+1, episodes, meanLoss, sigma0))
    
    Plist = np.array(Plist)
    if trace_P:
        return L, Plist
    return L
