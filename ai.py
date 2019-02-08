
# Importing Libraries

import numpy as np #Efficient use of arrays. np as shortcut
import random
import os # Load/Save model after running
import torch # Using PyTorch!
import torch.nn as nn# To implement Neural Network (nn)
import torch.nn.functional as F
import torch.optim as optim # Optimizers
import torch.autograd as autograd
from torch.autograd import Variable

# Starting Neural Network Layout (Architecture)
# Input Layer --> Hidden Layer --> Output Layer (Inside __init__)
# We use only one hidden layer here!
# TODO Add more layers / hidden neurons (30 now)

class Network(nn.Module):
    
    def __init__(self, input_size, nb_action): # Input_size is number of input neurons
                                              # nb_action is 3 outputs --> move forward, right, or left
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, 30)   # full_connection_1 --> connects input neurons to hidden layer (bias=true?)                                    
        self.fc2 = nn.Linear(30, nb_action)    # full_connection_2 --> connects hidden layer to output layer
        
    def forward(self, state):    # Takes state, performs actions, initiates neural network, returns Q values (raw returns only)
        
        x = F.relu(self.fc1(state))    # Activates hidden neurons with the help of fc1. fc1 connects input to hidden layers.
        q_values = self.fc2(x)    # Output neurons of nn, put into "q_values" which has all Q values. Will put into sortmax later
        return q_values