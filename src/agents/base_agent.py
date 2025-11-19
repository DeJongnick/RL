"""Base agent class defining the interface for all trading agents."""

from abc import ABC, abstractmethod
import numpy as np


class BaseAgent(ABC):
    """Abstract base class for all trading agents."""
    
    def __init__(self, action_space=None):
        """
        Initialize the base agent.
        
        Args:
            action_space: The action space from the environment (optional)
        """
        self.action_space = action_space
    
    @abstractmethod
    def act(self, state: np.ndarray, training: bool = False):
        """
        Choose an action given the current state.
        
        Args:
            state: Current state observation from the environment
            training: Whether the agent is in training mode
            
        Returns:
            action: The action to take (position value for trading agents)
        """
        pass
    
    def update(self, *args, **kwargs):
        """
        Update the agent's policy based on experience.
        
        This method is called during training to update the agent's parameters.
        For non-learning agents, this method can be a no-op.
        
        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        pass
    
    def save(self, filepath: str):
        """
        Save the agent's state to a file.
        
        Args:
            filepath: Path where to save the agent
        """
        pass
    
    def load(self, filepath: str):
        """
        Load the agent's state from a file.
        
        Args:
            filepath: Path from where to load the agent
        """
        pass
    
    def reset(self):
        """Reset agent state if needed (e.g., for new episode)."""
        pass

