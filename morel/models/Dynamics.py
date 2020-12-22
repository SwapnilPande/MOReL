import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicsNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons = 512, activation = nn.ReLU):
        super(DynamicsNet, self).__init__()


        # Validate inputs
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons

        # Create layers for the net
        self.input_layer = nn.Linear(input_dim, n_neurons)
        self.h0 = nn.Linear(n_neurons, n_neurons)
        self.h0_act = activation()
        self.h1 = nn.Linear(n_neurons, n_neurons)
        self.h1_act = activation()
        self.output_layer = nn.Linear(n_neurons, output_dim)


    def forward(self, x):
        x = self.input_layer(x)
        x = self.h0(x)
        x = self.h0_act(x)
        x = self.h1(x)
        x = self.h1_act(x)
        x = self.output_layer(x)

        return x

class DynamicsEnsemble():
    def __init__(self, input_dim, output_dim, n_models = 4, n_neurons = 512, n_layers = 2, activation = nn.ReLU):
        self.n_models = 4

        self.models = []

        for i in range(n_models):
            self.models.append(DynamicsNet(input_dim,
                                            output_dim,
                                            n_neurons = n_neurons,
                                            n_layers = n_layers,
                                            activation = nn.ReLu))

    def forward(self, model, x):
        return self.models[model](x)