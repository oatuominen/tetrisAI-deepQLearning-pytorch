import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np



class DeepQNetwork(nn.Module):

    def __init__(self, learning_rate):
        super(DeepQNetwork, self).__init__()
        self.learning_rate = learning_rate

        # fully connected layers
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss() 
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        journeys = self.fc4(x)
        return journeys

    def save_dqn(self):
        T.save(self.state_dict(), 'dqn.pth')

    def load_dqn(self):
        self.load_state_dict(T.load('dqn.pth'))
