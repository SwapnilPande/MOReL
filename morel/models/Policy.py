import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons = 32, activation = nn.Tanh):
        # Validate inputs
        assert n_layers > 0
        assert input_dim > 0
        assert output_dim > 0
        assert n_neurons > 0
        assert n_layers > 0

        # Store configuration parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_neurons = n_neurons
        self.n_layers = n_layers

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