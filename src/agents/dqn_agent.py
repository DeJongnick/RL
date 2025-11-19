"""
Deep Q-Network (DQN) Agent for Trading Environment

This agent implements a Deep Q-Learning algorithm with:
- Experience replay buffer
- Epsilon-greedy exploration
- Target network for stable learning
- Double DQN (uses main network to select actions, target network to evaluate)
- Dueling architecture (separates state value and action advantages)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import List, Optional, Any, Union, Dict
from pathlib import Path
import yaml

from ..models.dqn_network import DuelingDQN


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling transitions.
    
    This buffer stores (state, action, reward, next_state, done) tuples
    and allows random sampling to break correlation between consecutive experiences.
    """
    
    def __init__(self, capacity: int = 100000):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state observation
            action: Action index taken
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as numpy arrays
        """
        if len(self.buffer) < batch_size:
            raise ValueError(f"Buffer size ({len(self.buffer)}) is smaller than batch_size ({batch_size})")
        
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network agent for trading environments.
    
    This agent learns to select optimal positions (actions) by:
    - Using a neural network to approximate Q-values
    - Storing experiences in a replay buffer
    - Training on random batches of past experiences
    - Using epsilon-greedy exploration during training
    - Maintaining a target network for stable learning
    """
    
    def __init__(
        self,
        state_dim: int,
        positions: Optional[Union[List[float], str, Dict]] = None,
        config_path: Optional[str] = None,
        lr: float = 0.0001,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        buffer_size: int = 100000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        hidden_dim: int = 128,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space (7: 5 market features + 2 position features)
            positions: Can be:
                      - A list of available positions (e.g., [-1, 0, 0.5, 1, 2])
                      - A path to a YAML config file (string)
                      - A config dictionary
                      - None: will try to load from config_path or use defaults
            config_path: Path to YAML configuration file. If provided and positions
                        is None, will load positions from environment.positions in config.
            lr: Learning rate for the optimizer
            gamma: Discount factor for future rewards
            epsilon_start: Initial epsilon value for epsilon-greedy exploration
            epsilon_end: Final epsilon value (minimum exploration)
            epsilon_decay: Epsilon decay factor per episode
            buffer_size: Maximum size of the experience replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency of target network updates (in training steps)
            hidden_dim: Hidden layer dimension for the neural network
            device: PyTorch device (cuda or cpu). If None, auto-detects.
            seed: Random seed for reproducibility
        
        Position meanings:
            - -1: short position
            - 0: no position (hold cash)
            - 0.5: half position (long)
            - 1: full long position
            - 2: 2x leverage long position
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
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize networks
        self.q_network = DuelingDQN(state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_network = DuelingDQN(state_dim, self.action_dim, hidden_dim=hidden_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # Training step counter
        self.train_step_count = 0
    
    def _load_positions_from_config(self, config_path: str) -> List[float]:
        """
        Load positions from a YAML configuration file.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Returns:
            List of available positions from environment.positions
        
        Raises:
            FileNotFoundError: If the config file doesn't exist
            KeyError: If environment.positions is not found in config
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        return self._extract_positions_from_config(config)
    
    def _extract_positions_from_config(self, config: Dict) -> List[float]:
        """
        Extract positions from a configuration dictionary.
        
        Args:
            config: Configuration dictionary
        
        Returns:
            List of available positions from environment.positions
        
        Raises:
            KeyError: If environment.positions is not found in config
        """
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
        Select an action (position) using epsilon-greedy policy.
        
        During training, uses epsilon-greedy exploration:
        - With probability epsilon: selects a random position
        - Otherwise: selects the position with highest Q-value
        
        During evaluation, always selects the position with highest Q-value.
        
        Args:
            state: Current state observation (numpy array)
            training: Whether the agent is in training mode
        
        Returns:
            Selected position (float) from self.positions
        """
        # Epsilon-greedy exploration during training
        if training and random.random() < self.epsilon:
            return random.choice(self.positions)
        
        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            action_idx = q_values.argmax().item()
        
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
        Store a transition in the replay buffer.
        
        Args:
            state: Current state observation
            action: Position taken (will be converted to action index)
            reward: Reward received
            next_state: Next state observation
            done: Whether episode terminated
        """
        # Convert position to action index
        try:
            action_idx = self.positions.index(action)
        except ValueError:
            # If exact match not found, find closest position
            action_idx = min(range(len(self.positions)), 
                           key=lambda i: abs(self.positions[i] - action))
        
        self.replay_buffer.push(state, action_idx, reward, next_state, done)
    
    def update(self) -> Optional[float]:
        """
        Perform one training step by sampling from replay buffer and updating Q-network.
        
        This method:
        1. Samples a batch from the replay buffer
        2. Computes Q-values and target Q-values
        3. Updates the network using Double DQN
        4. Periodically updates the target network
        
        Returns:
            Training loss (float) if training occurred, None if buffer too small
        """
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q-values
        q_values = self.q_network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values using Double DQN
        # Use main network to select actions
        with torch.no_grad():
            next_q_values = self.q_network(next_states)
            next_actions = next_q_values.argmax(1)
            # Use target network to evaluate actions
            next_q_values_target = self.target_network(next_states)
            next_q_value = next_q_values_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self._update_target_network()
        
        return loss.item()
    
    def _update_target_network(self):
        """Copy weights from main network to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def decay_epsilon(self):
        """
        Decay epsilon for epsilon-greedy exploration.
        
        This should be called after each episode during training.
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str):
        """
        Save the agent's state (networks, optimizer, and hyperparameters).
        
        Args:
            filepath: Path where to save the agent
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step_count': self.train_step_count,
            'positions': self.positions,
            'state_dim': self.state_dim,
            'lr': self.lr,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
        }, filepath)
    
    def load(self, filepath: str):
        """
        Load the agent's state from a saved checkpoint.
        
        Args:
            filepath: Path from where to load the agent
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.epsilon = checkpoint.get('epsilon', self.epsilon_end)
        self.train_step_count = checkpoint.get('train_step_count', 0)
        
        # Set networks to evaluation mode
        self.q_network.eval()
        self.target_network.eval()
    
    def reset(self):
        """
        Reset agent state for a new episode.
        
        Note: This does not reset the replay buffer or learned weights,
        only episode-specific state if needed.
        """
        pass

