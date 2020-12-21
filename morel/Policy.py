import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim, n_neurons = 32, n_layers = 2, activation = nn.Tanh):
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
        self.layers = []

        # Add first hidden layer
        self.layers.append(nn.Linear(input_dim, n_neurons))
        if(activation is not None):
            self.layers.append(activation)

        # Add remaining hidden layers
        for i in range(n_layers - 1):
            self.layers.append(nn.Linear(n_neurons, n_neurons))
            if(activation is not None):
                self.layers.append(activation)

        # Add output layer
        self.layers.append(nn.Linear(n_neurons, output_dim))

    def forward(self, x):
        # Pass data through layers defined in self.layers
        for layer in self.layers:
            x = layer(x)

        return x