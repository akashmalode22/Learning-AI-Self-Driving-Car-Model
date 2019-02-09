
# Importing Libraries
# Learning AI from the online course "Artificial Intelligence A-Z Learn How To Build an AI"

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
    
# Experience Replay--------------------------------------------------------
        
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
    
        
# Deep Q Learning---------------------------------------------------------
        
class Dqn():  #Dqn --> Deep Q Network
    
    def __init__(self, input_size, nb_action, gamma):  # gamma --> Discount coefficient / Delay coeff (Bellman's equation)
        self.gamma = gamma
        self.reward_window = []    # Takes mean of the last 100 rewards, puts it in this reward_window --> Just to monitor evolution of rewards (how well AI is doing)
        self.model = Network(input_size, nb_action)    # Creates an nn model for the Dqn
        self.memory = ReplayMemory(100000)    # Taking 100,000 transitions for dqn to learn
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)    # Can choose any optimizer
        #Learning Rate (lr) should be small as possible, so the AI gets time to learn and learns from its mistakes (0.001 is good)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0    # Last action refers to the 3 rotations. last_action can be 0/1/2 refering to 0, 20, -20 rotation from the x-axis
        self.last_reward = 0
        
    # Function to make the car decide the right action at the right time
    
    def selectAction(self, state):    # State here is basically the input of the nn. nn takes in an input to give the output. The output is what we need, but the output is determined by the input, which is "state"
        # Here we implement the softmax function --> takes in all the generated Q values, and selects the best among all Q-values.
        # While taking the best action from softmax, we still have to continue exploring
        # Every Q value is an action --> up, right, or left. Every Q value has a probability of THAT action happening. We calculate the 
        # probabilities of all Q values, that add up to 1. SOFTMAX does all this.
        # Softmax --> gives the highest probability to the best Q value, so that there is a higher chance of the best Q value to take place (or execute)
        
        probs = F.softmax(self.model(Variable(state, volatile = True)) * 0)    # Saving memory, better performance (lecture)
        # '7' here is the temperature parameter
        # Temperature parameter just increases or decreases the probability of an action based on its final Q-values
        # Ex: softmax([1,2,3]) = [0.08,0.2,0.82]    now if we put *7, softmax([1,2,3]*7) = [0, 0.02, 0.99]
        # Here the lower probability actions lowered even more, and the higher ones increased more
        # Just to make sure the most probable one is the action that's taken the most
        
        # Now we get an action. action will be a random draw from the upper probs list.
        action = probs.multinomial() # This multinomial returns a pytorch with the fake dimension that we introduced before, so we convert it before returning anything
        return action.data[0,0] # -->>CHECK THIS<<--
    
    # Function to train dnn --> forward and backward propagation
    # Output, target, compare error from the output value and target value, back propagate the error value and update the apporpriate weight
    # The weight that is adjusted depends on which weight contributed to the output (which made the error) the most.
    #REFER TO STEP 11
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, torch.Tensor([batch_action.unsqueeze(1)])).squeeze(1)
        # Here we need to pass into the model only the action that was chosen from the dqn. gather(1, batch_action) takes only the actions
        # that were chosen in each state in the batch_state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward  # target function (basic cost function in deepAI)
        td_loss = F.smooth_l1_loss(outputs, target)
        # Re-initialize optimizer at start of each loop
        self.optimizer.zero_grad()
        td_loss.backward(retain_variables = True)
        # Update weights
        self.optimizer.step()
        
    # Connecting AI to map now
    
    def update(self, reward, new_signal):
        # First update the new state (everytime we enter a new state)
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)  # Again to take out the fake dimension we created previously
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # torch.LongTensor([int(self.last_action)]) --> way to convert simple integer into a torchTensor
        
        # Now we do an action
        action = self.selectAction(new_state)
        
        # After playing the action, we get a reward. Now we learn from it. AI learns from it's past 100 actions.
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
            
        # We now learned something new. Now to update our action
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        
        # Now put the reward into the reward window (so we can see if the AI is learning)
        
        self.reward_window.append(reward)
        if len(self.reward_window) < 1000:
            del self.reward_window[0]
            
        return action
    
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)    # +1 do avoid denominator from being zero
    
    # WE now make a save function --> we save only the neural network and the optimizer, because we need to save only the weight values
    def save(self):
        torch.save({'state_dict': self.model.state_dict(), 'optimizer': self.optimizer.state_dict()}, 'last_brain.pth')
        
    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("Loading...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done")
        else:
            print("File not found")
            
        
        
        
        
        
        
        
        
        
        


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    