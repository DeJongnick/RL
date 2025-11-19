"""Actor-Critic network architecture for PPO (Proximal Policy Optimization)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    The network has two heads:
    - Actor: outputs action probabilities (policy)
    - Critic: outputs state value estimate
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        Initialize the Actor-Critic network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of possible actions (discrete)
            hidden_dim: Size of hidden layers (default: 128)
        """
        super(ActorCritic, self).__init__()
        
        # Shared feature layers
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head: outputs logits for action probabilities
        self.actor_fc = nn.Linear(hidden_dim, action_dim)
        
        # Critic head: outputs state value
        self.critic_fc = nn.Linear(hidden_dim, 1)
    
    def forward(self, state):
        """
        Forward pass through the network.
        
        Args:
            state: State tensor of shape [batch_size, state_dim]
            
        Returns:
            action_logits: Logits for action probabilities, shape [batch_size, action_dim]
            value: State value estimate, shape [batch_size, 1]
        """
        # Shared feature extraction
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        # Actor: action logits
        action_logits = self.actor_fc(x)
        
        # Critic: state value
        value = self.critic_fc(x)
        
        return action_logits, value
    
    def get_action_and_value(self, state, action=None):
        """
        Get action distribution and value for a given state.
        Optionally compute log probability of a given action.
        
        Args:
            state: State tensor
            action: Optional action index to compute log prob for
            
        Returns:
            action: Sampled action (if action is None)
            action_log_prob: Log probability of the action
            entropy: Entropy of the action distribution
            value: State value estimate
        """
        action_logits, value = self.forward(state)
        action_probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        
        if action is None:
            action = dist.sample()
        
        action_log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, action_log_prob, entropy, value

