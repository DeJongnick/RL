"""
Proximal Policy Optimization (PPO) Agent for Trading Environment

This agent implements PPO with:
- Actor-Critic architecture
- Clipped surrogate objective
- Generalized Advantage Estimation (GAE)
- Multiple epochs of updates per batch
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
import yaml

from ..models.ppo_network import ActorCritic


class RolloutBuffer:
    """
    Buffer for storing rollout trajectories for PPO.
    
    Stores (state, action, reward, value, log_prob, done) tuples
    and computes advantages using GAE (Generalized Advantage Estimation).
    """
    
    def __init__(self, gamma=0.99, gae_lambda=0.95):
        """
        Initialize the rollout buffer.
        
        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        
    def add(self, state, action, reward, value, log_prob, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate
            log_prob: Log probability of the action
            done: Whether episode terminated
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
    
    def compute_advantages_and_returns(self, last_value=0.0, last_done=False):
        """
        Compute advantages and returns using GAE.
        
        Args:
            last_value: Value estimate for the last state (if episode didn't terminate)
            last_done: Whether the last state was terminal
            
        Returns:
            advantages: Computed advantages
            returns: Computed returns (advantages + values)
        """
        advantages = []
        returns = []
        
        gae = 0
        next_value = last_value
        next_done = last_done
        
        # Compute advantages backwards
        for step in reversed(range(len(self.rewards))):
            if step == len(self.rewards) - 1:
                # Last step
                delta = self.rewards[step] + self.gamma * next_value * (1 - next_done) - self.values[step]
                gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * gae
            else:
                delta = self.rewards[step] + self.gamma * self.values[step + 1] * (1 - self.dones[step]) - self.values[step]
                gae = delta + self.gamma * self.gae_lambda * (1 - self.dones[step]) * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + self.values[step])
        
        return np.array(advantages), np.array(returns)
    
    def get_batch(self):
        """
        Get all stored data as numpy arrays.
        
        Returns:
            Tuple of (states, actions, old_log_probs, advantages, returns, values)
        """
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            np.array(self.values)
        )
    
    def clear(self):
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for trading environments.
    
    This agent learns a policy by:
    - Collecting rollouts of experiences
    - Computing advantages using GAE
    - Updating the policy using clipped PPO objective
    - Updating the value function using MSE loss
    """
    
    def __init__(
        self,
        state_dim: int,
        positions: Optional[Union[List[float], str, Dict]] = None,
        config_path: Optional[str] = None,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 4,
        batch_size: int = 64,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            positions: Can be:
                      - A list of available positions (e.g., [-1, 0, 0.5, 1, 2])
                      - A path to a YAML config file (string)
                      - A config dictionary
                      - None: will try to load from config_path or use defaults
            config_path: Path to YAML configuration file
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_coef: Coefficient for value loss
            entropy_coef: Coefficient for entropy bonus
            max_grad_norm: Maximum gradient norm for clipping
            update_epochs: Number of epochs to update on each batch
            batch_size: Batch size for updates
            hidden_dim: Hidden layer dimension for the neural network
            device: PyTorch device (cuda or cpu). If None, auto-detects.
            seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Load positions from config if needed
        if positions is None:
            if config_path is not None:
                positions = self._load_positions_from_config(config_path)
            else:
                # Default positions
                positions = [-1, 0, 0.5, 1, 2]
        elif isinstance(positions, str):
            # positions is a path to config file
            positions = self._load_positions_from_config(positions)
        elif isinstance(positions, dict):
            # positions is a config dictionary
            positions = self._extract_positions_from_config(positions)
        
        self.positions = positions
        self.action_dim = len(positions)
        self.state_dim = state_dim
        
        # Hyperparameters
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize network
        self.network = ActorCritic(state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        
        # Optimizer
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        
        # Rollout buffer
        self.rollout_buffer = RolloutBuffer(gamma=gamma, gae_lambda=gae_lambda)
        
        # Training step counter
        self.train_step_count = 0
    
    def _load_positions_from_config(self, config_path: str) -> List[float]:
        """Load positions from a YAML configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._extract_positions_from_config(config)
    
    def _extract_positions_from_config(self, config: Dict) -> List[float]:
        """Extract positions from a configuration dictionary."""
        if 'environment' not in config:
            raise KeyError("'environment' key not found in configuration")
        
        if 'positions' not in config['environment']:
            raise KeyError("'positions' key not found in environment configuration")
        
        positions = config['environment']['positions']
        
        if not isinstance(positions, list):
            raise ValueError("'positions' must be a list")
        
        return positions
    
    def act(self, state: np.ndarray, training: bool = False) -> float:
        """
        Select an action (position) using the current policy.
        
        Args:
            state: Current state observation (numpy array)
            training: Whether the agent is in training mode (affects exploration)
        
        Returns:
            Selected position (float) from self.positions
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.network.get_action_and_value(state_tensor)
            action_idx = action.item()
        
        return self.positions[action_idx]
    
    def store_transition(
        self,
        state: np.ndarray,
        action: float,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store a transition in the rollout buffer.
        
        Args:
            state: Current state observation
            action: Position taken (will be converted to action index)
            reward: Reward received
            next_state: Next state observation (not used in PPO, but kept for interface compatibility)
            done: Whether episode terminated
        """
        # Convert position to action index
        try:
            action_idx = self.positions.index(action)
        except ValueError:
            # If exact match not found, find closest position
            action_idx = min(range(len(self.positions)),
                           key=lambda i: abs(self.positions[i] - action))
        
        # Get value and log prob for current state
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, log_prob, _, value = self.network.get_action_and_value(
                state_tensor, action=torch.tensor([action_idx]).to(self.device)
            )
        
        self.rollout_buffer.add(
            state=state,
            action=action_idx,
            reward=reward,
            value=value.item(),
            log_prob=log_prob.item(),
            done=done
        )
    
    def update(self, last_value: float = 0.0, last_done: bool = False) -> Optional[Dict[str, float]]:
        """
        Perform PPO update on the collected rollouts.
        
        Args:
            last_value: Value estimate for the last state (if episode didn't terminate)
            last_done: Whether the last state was terminal
            
        Returns:
            Dictionary with training metrics (loss, policy_loss, value_loss, entropy) or None if buffer is empty
        """
        if len(self.rollout_buffer.states) == 0:
            return None
        
        # Compute advantages and returns
        advantages, returns = self.rollout_buffer.compute_advantages_and_returns(
            last_value=last_value, last_done=last_done
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get batch data
        states, actions, old_log_probs, old_values = self.rollout_buffer.get_batch()
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        old_values = torch.FloatTensor(old_values).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        
        # Training metrics
        total_loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        entropy_loss = 0.0
        
        # Multiple epochs of updates
        for epoch in range(self.update_epochs):
            # Shuffle data
            indices = torch.randperm(len(states))
            
            # Mini-batch updates
            for start in range(0, len(states), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_values = old_values[batch_indices]
                
                # Get current policy predictions
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    batch_states, action=batch_actions
                )
                
                # Compute policy loss (PPO clipped objective)
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss_batch = -torch.min(surr1, surr2).mean()
                
                # Compute value loss
                value_loss_batch = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Compute entropy loss (we want to maximize entropy, so negate it)
                entropy_loss_batch = -entropy.mean()
                
                # Total loss
                loss = policy_loss_batch + self.value_coef * value_loss_batch + self.entropy_coef * entropy_loss_batch
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Accumulate metrics
                total_loss += loss.item()
                policy_loss += policy_loss_batch.item()
                value_loss += value_loss_batch.item()
                entropy_loss += entropy_loss_batch.item()
        
        # Average metrics
        num_updates = self.update_epochs * (len(states) // self.batch_size + (1 if len(states) % self.batch_size else 0))
        metrics = {
            'loss': total_loss / num_updates,
            'policy_loss': policy_loss / num_updates,
            'value_loss': value_loss / num_updates,
            'entropy': -entropy_loss / num_updates  # Convert back to positive entropy
        }
        
        # Clear buffer
        self.rollout_buffer.clear()
        
        self.train_step_count += 1
        
        return metrics
    
    def save(self, filepath: str):
        """
        Save the agent's state (network, optimizer, and hyperparameters).
        
        Args:
            filepath: Path where to save the agent
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_step_count': self.train_step_count,
            'positions': self.positions,
            'state_dim': self.state_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'gae_lambda': self.gae_lambda,
            'clip_epsilon': self.clip_epsilon,
            'value_coef': self.value_coef,
            'entropy_coef': self.entropy_coef,
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load the agent's state from a saved checkpoint.
        
        Args:
            filepath: Path from where to load the agent
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_step_count = checkpoint.get('train_step_count', 0)
        
        # Set network to evaluation mode
        self.network.eval()
    
    def reset(self):
        """
        Reset agent state for a new episode.
        
        Clears the rollout buffer.
        """
        self.rollout_buffer.clear()

