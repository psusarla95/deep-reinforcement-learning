import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """
    Actor POlicy Model (mu)
    """

    def __init__(self, state_size, action_size, seed, fc1=128, fc2=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=fc1)
        self.fc2 = nn.Linear(in_features=fc1, out_features=fc2)
        self.fc3 = nn.Linear(in_features=fc2, out_features=action_size)
        #self.fc4 = nn.Linear(in_features=fc3, out_features=action_size)
        self.bn1 = nn.BatchNorm1d(fc1)
        self.reset_parameters()


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        #self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tensor):
        """
        Build an actor policy network that maps states to actions
        """

        #input layer
        t = state_tensor

        # (1) hidden linear layer
        t = self.fc1(t)
        t = F.relu(t)
        t = self.bn1(t)

        # (2) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (3) Output layer
        t = self.fc3(t)
        t = F.tanh(t)

        return t


class Critic(nn.Module):

    """
    Build a Critic Value network
    """

    def __init__(self, state_size, action_size, seed, fcs1_units=128, fc2_units=128):
        """Initialize parameters and build model.
            Params
            ======
                state_size (int): Dimension of each state
                action_size (int): Dimension of each action
                seed (int): Random seed
                fcs1_units (int): Number of nodes in the first hidden layer
                fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fcs1 = nn.Linear(in_features=state_size, out_features=fcs1_units)
        self.fc2 = nn.Linear(in_features=fcs1_units+action_size, out_features=fc2_units)
        self.fc3 = nn.Linear(in_features=fc2_units,out_features=1)

        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.reset_parameters()

    def reset_parameters(self):
        self.fcs1.weight.data.uniform_(*hidden_init(self.fcs1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state_tensor, action_tensor):

        xs = F.relu(self.fcs1(state_tensor))
        xs = self.bn1(xs)
        x = torch.cat((xs, action_tensor), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
