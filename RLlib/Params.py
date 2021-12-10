import torch.optim as optim


class Constants():
    constants = {}
    # The Dynamics of the System
    constants['L'] = 1.
    constants['g'] = 9.82
    constants['m'] = 1.
    constants['my'] = 0.01

    # These are constants that influence the agents
    constants['R'] = 0.04
    constants['tau'] = 1.
    constants['uMax'] = 5
    constants['wMax'] = 1000
    constants['gamma_d'] = 0.22

    # Constants for learning
    constants['dt'] = 0.02
    constants['gamma'] = 1 - constants['dt']/constants['tau']
    constants['kappa'] = 0.1
    constants['lambda'] = (1 - constants['dt']/constants['kappa'])/constants['gamma']
    constants['C_lr'] = 0.1
    constants['A_lr'] = 0.1
    constants['optimizer'] = "SGD"  # "Adam" or "SGD"

    # Constants for noise in learning process
    constants['tau_n'] = 0.25
    constants['sigma0'] = 0.5
    constants['noise0'] = 0
    constants['seed'] = None  # None to disable, any integer to enable

    bundle = {}
    bundle['Dynamics'] = ['L', 'g', 'm', 'my']
    bundle['Agents'] = ['R', 'tau', 'gamma_d', 'uMax', 'wMax']
    bundle['Learning'] = ['dt', 'gamma', 'kappa', 'lambda', 'C_lr',
                          'A_lr', 'optimizer']
    bundle['Noise'] = ['tau_n', 'sigma0', 'noise0', 'seed']

    def __init__(self):
        pass

    def get(self, keyword):
        return self.constants[keyword]

    def add(self, keyword, value):
        self.constants[keyword] = value

    def __str__(self):
        out = ""
        for bundle_key in self.bundle:
            out += "  Constants that affect {0}\n".format(bundle_key)
            for key in self.bundle[bundle_key]:
                out += key + " "*(10 - len(key)) + str(self.constants[key])+"\n"
            out += "\n"
        return out


def optim_dict(agent, lr, keyword):
    if keyword == "SGD":
        return optim.SGD(agent.parameters, lr=lr)
    elif keyword == "Adam":
        return optim.Adam(agent.parameters, lr=lr)
