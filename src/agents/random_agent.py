"""
Random Agent for Trading Environment

This agent takes random positions on the market without any learning or strategy.
It serves as a baseline for comparison with more sophisticated agents.
"""

import random
import numpy as np
import yaml
from pathlib import Path
from typing import List, Optional, Any, Union, Dict


class RandomAgent:
    """
    An agent that selects random positions from the available action space.
    
    This agent is useful as a baseline to compare the performance of
    more sophisticated trading strategies.
    """
    
    def __init__(
        self,
        positions: Optional[Union[List[float], str, Dict]] = None,
        config_path: Optional[str] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize the random agent.
        
        Args:
            positions: Can be:
                      - A list of available positions (e.g., [-1, 0, 0.5, 1, 2])
                      - A path to a YAML config file (string)
                      - A config dictionary
                      - None: will try to load from config_path or use defaults
            config_path: Path to YAML configuration file. If provided and positions
                        is None, will load positions from environment.positions in config.
                        If positions is provided, this parameter is ignored.
            seed: Random seed for reproducibility
        
        Position meanings:
            - -1: short position
            - 0: no position (hold cash)
            - 0.5: half position (long)
            - 1: full long position
            - 2: 2x leverage long position
        """
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
        self.action_space_size = len(positions)
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
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
    
    def act(self, state: Any, training: bool = False) -> float:
        """
        Select a random action (position) from the available positions.
        
        Args:
            state: Current state of the environment (not used by random agent)
            training: Whether the agent is in training mode (not used by random agent)
        
        Returns:
            A random position from the available positions
        """
        return random.choice(self.positions)
    
    def update(self, *args, **kwargs):
        """
        Update method for compatibility with training loops.
        Random agent doesn't learn, so this method does nothing.
        
        Args:
            *args: Variable length argument list (ignored)
            **kwargs: Arbitrary keyword arguments (ignored)
        """
        pass
    
    def save(self, filepath: str):
        """
        Save agent state (for compatibility).
        Random agent has no state to save.
        
        Args:
            filepath: Path where to save the agent
        """
        pass
    
    def load(self, filepath: str):
        """
        Load agent state (for compatibility).
        Random agent has no state to load.
        
        Args:
            filepath: Path from where to load the agent
        """
        pass
    
    def reset(self):
        """
        Reset agent state (for compatibility).
        Random agent has no state to reset.
        """
        pass

