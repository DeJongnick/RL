"""Utilities for creating training and evaluation environments."""

import gymnasium as gym
import gym_trading_env  # Ensure custom environment is registered


def create_training_env(
    df_train, 
    positions=[-1, 0, 0.5, 1, 2], 
    trading_fees=0.01/100, 
    borrow_interest_rate=0.0003/100
):
    """
    Create a gym environment for training.
    
    Args:
        df_train: Training DataFrame with market data
        positions: List of available position values
        trading_fees: Trading fee rate (as decimal, e.g., 0.01/100 = 0.01%)
        borrow_interest_rate: Borrow interest rate per timestep
    
    Returns:
        Gymnasium environment for training
    """
    return gym.make(
        "TradingEnv",
        name="BTCUSD",
        df=df_train,
        positions=positions,
        trading_fees=trading_fees,
        borrow_interest_rate=borrow_interest_rate,
    )


def create_eval_env(
    df_eval,
    positions=[-1, 0, 0.5, 1, 2],
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100
):
    """
    Create a gym environment for evaluation.
    
    Args:
        df_eval: Evaluation DataFrame with market data
        positions: List of available position values
        trading_fees: Trading fee rate (as decimal, e.g., 0.01/100 = 0.01%)
        borrow_interest_rate: Borrow interest rate per timestep
    
    Returns:
        Gymnasium environment for evaluation
    """
    return gym.make(
        "TradingEnv",
        name="BTCUSD",
        df=df_eval,
        positions=positions,
        trading_fees=trading_fees,
        borrow_interest_rate=borrow_interest_rate,
    )

