import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(in_features=state_size, out_features=256)
        nn.init.xavier_uniform_(self.fc1.weight)
        #nn.init.zeros_(self.fc1.bias)

        self.fc2 = nn.Linear(in_features=256, out_features=256)
        nn.init.xavier_uniform_(self.fc2.weight)
        #nn.init.zeros_(self.fc1.bias)

        self.fc3 = nn.Linear(in_features=256, out_features=128)
        nn.init.xavier_uniform_(self.fc3.weight)
        #self.fc2.bias.data.zeros_()

        self.out = nn.Linear(in_features=128, out_features=action_size)
        nn.init.xavier_uniform_(self.out.weight)
        #self.out.bias.data.zeros_()


    def forward(self, state_tensor):
        """Build a network that maps state -> action values."""
        # pass
        # forward through each layer in "hidden layer", with ReLU activation unit between them

        #input layer
        t = state_tensor

        # (1) hidden linear layer
        t = self.fc1(t)
        t = F.relu(t)

        # (2) hidden linear layer
        t = self.fc2(t)
        t = F.relu(t)

        # (3) hidden linear layer
        t = self.fc3(t)
        t = F.relu(t)

        # (4) Output layer
        t = self.out(t)
        return t