
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
    
# Experience Replay
        
class ReplayMemory(object):
    
    def __init__(self, capacity):    # Capacity here is number of memory keeps (number of previous transitions - 100)
        self.capacity = capacity
        self.memory = []    # Transitions are stored in this memory list of capacity "capacity"
        
    def push(self, event):      # "event" is put into "memory" list. "event" has --> last state, new state, last action, last reward
        self.memory.append(event)
        if len(self.memory) > self.capacity:       # To limit memory to have "capacity" elements
            del self.memory[0]
            
    # At this point, we have random "capacity" number of samples from test cases stored inside "memory" list
    # Now we create a function to extract random samples from the "memory" list
    
    def sample(self, batch_size):
        # zip(*) reshapes list
        # list = ((1,2,3),(4,5,6)) --> zip(*list) = ((1,4),(2,5),(3,6))
        # memory stores here as (state, action, reward), (state, action, reward), (state, action, reward)
        # zip of this gives --> (state, state, state), (action, action, action), (reward, reward, reward)
        # We use zip to re-organize all states together, all actions together, all rewards together
        samples = zip(*random.sample(self.memory, batch_size))
        # Put samples into a PyTorch variable (--->>CHECK THIS<<---)
        return map(lambda x: Variable(torch.cat(x, 0)), samples)  # Tensors are mult-dimensional matrix store values (0 dimensions in this case????)
    
        
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    