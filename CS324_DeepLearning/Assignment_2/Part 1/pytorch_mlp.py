from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes, dropout=0):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
            dropout: dropout probability
        """
        super(MLP, self).__init__()
        # Define the layers of the network
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        # Add input layer
        self.layers.append(nn.Linear(n_inputs, n_hidden[0]))
        self.dropouts.append(nn.Dropout(dropout))
        # Add hidden layers
        for i in range(1, len(n_hidden)):
            self.layers.append(nn.Linear(n_hidden[i-1], n_hidden[i]))
            self.dropouts.append(nn.Dropout(dropout))
        # Add output layer
        self.layers.append(nn.Linear(n_hidden[-1], n_classes))

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        # Forward pass
        out = x
        for layer, dropout in zip(self.layers[:-1], self.dropouts):
            out = F.relu(layer(out))
            out = dropout(out)
        out = self.layers[-1](out)

        return out