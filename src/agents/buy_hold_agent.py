"""
Buy and Hold Agent for Trading Environment

This agent implements a simple buy-and-hold strategy:
- Takes a long position at the beginning
- Maintains that position throughout the entire episode
- Never sells or changes position

This serves as a baseline strategy to compare with more sophisticated agents.
"""

import yaml
from pathlib import Path
from typing import List, Optional, Any, Union, Dict


class BuyHoldAgent:
    """
    An agent that implements a buy-and-hold strategy.
    
    This agent takes a long position (typically position 1.0) at the start
    and maintains it throughout the entire trading period without any changes.
    """
    
    def __init__(
        self,
        position: Optional[float] = None,
        positions: Optional[Union[List[float], str, Dict]] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the buy-and-hold agent.
        
        Args:
            position: The position to hold (e.g., 1.0 for full long position).
                     If None, will try to determine from positions or config.
            positions: Can be:
                      - A list of available positions (e.g., [-1, 0, 0.5, 1, 2])
                      - A path to a YAML config file (string)
                      - A config dictionary
                      - None: will try to load from config_path or use defaults
            config_path: Path to YAML configuration file. If provided and positions
                        is None, will load positions from environment.positions in config.
                        If position is provided, this parameter is ignored.
        
        Position meanings:
            - -1: short position
            - 0: no position (hold cash)
            - 0.5: half position (long)
            - 1: full long position (default for buy-and-hold)
            - 2: 2x leverage long position
        """
        # Determine the position to hold
        if position is not None:
            self.position = position
        else:
            # Try to get positions from config or use defaults
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
            
            # For buy-and-hold, prefer full long position (1.0)
            # If not available, use the largest positive position
            if 1.0 in positions:
                self.position = 1.0
            else:
                # Find the largest positive position
                positive_positions = [p for p in positions if p > 0]
                if positive_positions:
                    self.position = max(positive_positions)
                else:
                    # Fallback to 0 if no positive positions available
                    self.position = 0
        
        self.initialized = False
    
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
        Select the buy-and-hold action (position).
        
        This method always returns the same position that was determined
        during initialization, implementing the buy-and-hold strategy.
        
        Args:
            state: Current state of the environment (not used by buy-and-hold agent)
            training: Whether the agent is in training mode (not used by buy-and-hold agent)
        
        Returns:
            The position to hold (typically 1.0 for full long position)
        """
        self.initialized = True
        return self.position
    
    def update(self, *args, **kwargs):
        """
        Update method for compatibility with training loops.
        Buy-and-hold agent doesn't learn, so this method does nothing.
        
        Args:
            *args: Variable length argument list (ignored)
            **kwargs: Arbitrary keyword arguments (ignored)
        """
        pass
    
    def save(self, filepath: str):
        """
        Save agent state (for compatibility).
        Buy-and-hold agent has minimal state to save.
        
        Args:
            filepath: Path where to save the agent
        """
        import pickle
        state = {
            'position': self.position,
            'initialized': self.initialized
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load(self, filepath: str):
        """
        Load agent state (for compatibility).
        
        Args:
            filepath: Path from where to load the agent
        """
        import pickle
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.position = state.get('position', 1.0)
        self.initialized = state.get('initialized', False)
    
    def reset(self):
        """
        Reset agent state (for compatibility).
        Buy-and-hold agent maintains the same position, but resets initialization flag.
        """
        self.initialized = False

