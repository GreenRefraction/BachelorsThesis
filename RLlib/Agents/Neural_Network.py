import torch
import pickle


class Neural_Network(torch.nn.Module):
    def __init__(self):
        super(Neural_Network, self).__init__()
        self.layers = []
        self.Funcs = []
        self.myParameters = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cuda(device)

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if self.Funcs[i] is not None:
                x = self.Funcs[i](x)
        return x

    def add(self, Size, activation, input_size=None):
        if input_size is None:
            input_size = self.layers[-1].out_features
        self.layers.append(torch.nn.Linear(input_size, Size))
        self.Funcs.append(activation)
        self.myParameters.append(self.layers[-1].weight)
        self.myParameters.append(self.layers[-1].bias)

    def Save(self, path):
        pickle_out = open(path+'.pickle', 'wb')
        pickle.dump(self, pickle_out)
        pickle_out.close()

    def Load(self, path):
        pickle_in = open(path + '.pickle', 'rb')
        temp = pickle.load(pickle_in)
        self.layers = temp.layers
        self.Funcs = temp.Funcs
        self.myParameters = temp.myParameters
