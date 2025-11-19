"""Dueling DQN network architecture for Deep Q-Learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture that separates state value and action advantages.
    
    The network outputs Q-values by combining:
    - V(s): State value function
    - A(s,a): Action advantage function
    
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Dueling DQN network.
        
        Args:
            state_dim: Dimension of the state space (7: 5 market features + 2 position features)
            action_dim: Number of possible actions (5 positions)
            hidden_dim: Size of hidden layers (default: 128)
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Value stream: estimates V(s)
        self.value_fc1 = nn.Linear(hidden_dim, 64)
        self.value_fc2 = nn.Linear(64, 1)
        
        # Advantage stream: estimates A(s,a)
        self.advantage_fc1 = nn.Linear(hidden_dim, 64)
        self.advantage_fc2 = nn.Linear(64, action_dim)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape [batch_size, state_dim]
            
        Returns:
            q_values: Q-values for each action, shape [batch_size, action_dim]
        """
        # Shared feature extraction
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
