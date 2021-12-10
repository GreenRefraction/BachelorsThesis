import numpy as np
from .Params import Constants


class NoiseClass(Constants):
    def __init__(self):
        super(NoiseClass, self).__init__()
        self.tau_n = self.get('tau_n')
        self.dt = self.get('dt')
        self.seed = self.get('seed')
        np.random.seed(self.seed)

    def LPFNNoise(self, noise):
        """Low pass filtered normally distributed 0, 1 noise"""
        dt = self.dt
        tau_n = self.tau_n
        Nnoise = np.random.normal(0, 1)
        noise = noise * (1 - dt/tau_n) + dt/tau_n * Nnoise
        return noise
