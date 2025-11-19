"""Visualization utilities for training, evaluation, and decision making."""

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")


def _to_float(val):
    """Convert value to float, returning None if conversion fails."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _first_present(d: Optional[dict], keys: Iterable[str], default=None):
    """Return the first present key value from dictionary."""
    if not isinstance(d, dict):
        return default
    for k in keys:
        v = d.get(k)
        if v is not None:
            return v
    return default


def _get_env_attr(env, names: Iterable[str]):
    """Get the first present attribute from environment."""
    for n in names:
        v = getattr(env, n, None)
        if v is not None:
            return v
    return None


def plot_training_curves(csv_path, save_path=None):
    """
    Plot training curves from training log CSV.
    
    Args:
        csv_path: Path to training log CSV file
        save_path: Path to save the plot (None to display)
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if 'episode' not in df:
        df['episode'] = np.arange(1, len(df) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes[0,0].plot(df['episode'], df.get('reward', pd.Series([0]*len(df))), label='Reward')
    axes[0,0].set(title='Episode Reward', xlabel='Episode', ylabel='Reward')
    if 'q_value' in df: 
        axes[0,1].plot(df['episode'], df['q_value'], label='Q-Value', color='orange')
    axes[0,1].set(title='Average Q-Value', xlabel='Episode', ylabel='Q-Value')
    if 'loss' in df: 
        axes[1,0].plot(df['episode'], df['loss'], label='Loss', color='green')
    axes[1,0].set(title='Average Training Loss', xlabel='Episode', ylabel='Loss')
    if 'epsilon' in df: 
        axes[1,1].plot(df['episode'], df['epsilon'], label='Epsilon', color='purple')
    axes[1,1].set(title='Epsilon Decay', xlabel='Episode', ylabel='Epsilon')
    for ax in axes.flat: 
        ax.legend(loc='best')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def plot_evaluation_results(df_results, save_path=None):
    """
    Evaluation results: portfolio vs market, return distributions, etc.
    
    Args:
        df_results: DataFrame with evaluation results
        save_path: Path to save the plot (None to display)
    """
    fig, axes = plt.subplots(2, 2, figsize=(15,10))
    episodes = df_results['episode']
    # Portfolio and market returns
    axes[0,0].plot(episodes, df_results['portfolio_return']*100, 'b-o', label='Portfolio', markersize=6)
    axes[0,0].plot(episodes, df_results['market_return']*100, 'r--s', label='Market', markersize=6)
    axes[0,0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,0].set(xlabel='Episode', ylabel='Return (%)', title='Portfolio vs Market Returns')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    # Excess return
    axes[0,1].bar(episodes, df_results['excess_return']*100, color='green', alpha=0.7)
    axes[0,1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0,1].set(xlabel='Episode', ylabel='Excess Return (%)', title='Excess Return (Portfolio - Market)')
    axes[0,1].grid(True, alpha=0.3)
    # Return distributions
    axes[1,0].hist(df_results['portfolio_return']*100, bins=20, alpha=0.7, label='Portfolio', color='blue')
    axes[1,0].hist(df_results['market_return']*100, bins=20, alpha=0.7, label='Market', color='red')
    axes[1,0].set(xlabel='Return (%)', ylabel='Frequency', title='Return Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    # Total reward
    axes[1,1].plot(episodes, df_results['total_reward'], 'g-o', markersize=6)
    axes[1,1].set(xlabel='Episode', ylabel='Total Reward', title='Total Reward per Episode')
    axes[1,1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def plot_trading_decisions(env, agent, save_path=None, max_steps=None, episode_data=None):
    """
    Plot the sequence of trading decisions for one episode.
    
    Args:
        env: Trading environment
        agent: Trading agent
        save_path: Path to save the plot (None to display)
        max_steps: Maximum steps to collect
        episode_data: Pre-collected episode data (optional)
    """
    if episode_data is None:
        episode_data = collect_episode_data(env, agent, max_steps=max_steps)
    steps = episode_data.get('steps', [])
    if not steps:
        raise ValueError("No episode data found to plot decisions.")
    indices = [s['step'] for s in steps]
    positions = [s.get('position', s.get('action', 0)) for s in steps]
    prices = [s.get('price') for s in steps]
    fig, axes = plt.subplots(2, 1, figsize=(15,10), sharex=True)
    axes[0].plot(indices, positions, 'o-', markersize=4, label='Position')
    axes[0].set(ylabel='Position', title='Trading Decisions Over Time')
    axes[0].legend()
    if any(p is not None for p in prices):
        ax2 = axes[0].twinx()
        ax2.plot(indices, [p if p is not None else np.nan for p in prices], color='gray', alpha=0.5, label='Price')
        ax2.set_ylabel('Price')
        ax2.legend(loc='lower right')
    portfolio_values = [s.get('portfolio_value') for s in steps]
    if any(v is not None for v in portfolio_values):
        axes[1].plot(indices, [v if v is not None else np.nan for v in portfolio_values], 'g-', label='Portfolio Value')
        axes[1].set_ylabel('Portfolio Value')
    else:
        axes[1].plot(indices, [s.get('cumulative_reward') for s in steps], 'g-', label='Cumulative Reward')
        axes[1].set_ylabel('Cumulative Reward')
    axes[1].set(xlabel='Step', title='Performance Over Time')
    axes[1].legend()
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def plot_baseline_comparison(results_dict, save_path=None):
    """
    Compare the RL agent's performance with baselines.
    
    Args:
        results_dict: Dictionary mapping agent names to result DataFrames
        save_path: Path to save the plot (None to display)
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    agent_names = list(results_dict)
    avg_portfolio_returns = [df['portfolio_return'].mean()*100 for df in results_dict.values()]
    avg_market_returns = [df['market_return'].mean()*100 for df in results_dict.values()]
    x = np.arange(len(agent_names))
    width = 0.35
    axes[0].bar(x-width/2, avg_portfolio_returns, width, label='Portfolio Return', alpha=0.8)
    axes[0].bar(x+width/2, avg_market_returns, width, label='Market Return', alpha=0.8)
    axes[0].set(xlabel='Agent', ylabel='Average Return (%)', title='Average Returns Comparison')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(agent_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    avg_excess_returns = [df['excess_return'].mean()*100 for df in results_dict.values()]
    colors = ['green' if r > 0 else 'red' for r in avg_excess_returns]
    axes[1].bar(agent_names, avg_excess_returns, color=colors, alpha=0.7)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].set(xlabel='Agent', ylabel='Average Excess Return (%)', title='Average Excess Return Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def plot_portfolio_performance(episode_data, save_path=None):
    """
    Plot portfolio value/cumulative reward and stepwise rewards.
    
    Args:
        episode_data: Dictionary with episode step data
        save_path: Path to save the plot (None to display)
    """
    steps = episode_data.get('steps', [])
    if not steps:
        raise ValueError("No episode data found to plot portfolio performance.")
    indices = [s['step'] for s in steps]
    portfolio_values = [s.get('portfolio_value') for s in steps]
    cumulative_rewards = [s.get('cumulative_reward') for s in steps]
    rewards = [s.get('reward', 0.0) for s in steps]
    fig, axes = plt.subplots(2, 1, figsize=(15,10), sharex=True)
    if any(v is not None for v in portfolio_values):
        axes[0].plot(indices, [v if v is not None else np.nan for v in portfolio_values], label='Portfolio Value')
        axes[0].set(ylabel='Value', title='Portfolio Value Over Time')
    else:
        axes[0].plot(indices, cumulative_rewards, label='Cumulative Reward')
        axes[0].set(ylabel='Reward', title='Cumulative Reward Over Time')
    axes[0].legend()
    axes[1].bar(indices, rewards, color='orange', alpha=0.7, label='Step Reward')
    axes[1].set(xlabel='Step', ylabel='Reward', title='Step Rewards')
    axes[1].legend()
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def plot_trading_actions_timeline(episode_data, env=None, save_path=None):
    """
    Timeline of agent actions for a single episode.
    
    Args:
        episode_data: Dictionary with episode step data
        env: Trading environment (optional)
        save_path: Path to save the plot (None to display)
    """
    steps = episode_data.get('steps', [])
    if not steps:
        raise ValueError("No episode data found for action timeline.")
    indices = [s['step'] for s in steps]
    actions = [s.get('action', s.get('position', 0)) for s in steps]
    fig, ax = plt.subplots(figsize=(15,4))
    scatter = ax.scatter(indices, actions, c=actions, cmap='viridis', s=30)
    ax.plot(indices, actions, alpha=0.3, color='gray')
    ax.set(xlabel='Step', ylabel='Action/Position', title='Trading Actions Timeline')
    fig.colorbar(scatter, ax=ax, label='Action')
    plt.tight_layout()
    if save_path: 
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else: 
        plt.show()
    plt.close()


def collect_episode_data(env, agent, max_steps=None):
    """
    Step through an episode and collect key information at every step.
    
    Args:
        env: Trading environment
        agent: Trading agent (must have act() method)
        max_steps: Maximum steps to collect
    
    Returns:
        Dictionary with episode data
    """
    obs, info = env.reset()
    done, truncated = False, False
    steps = []
    cumulative_reward, step_idx = 0.0, 0
    
    # Get positions from environment or agent
    positions = getattr(env, 'positions', None) or getattr(agent, 'positions', None)

    while not done and not truncated:
        # Agent returns position (float)
        position = agent.act(obs, training=False)
        
        # Convert position to action index if needed
        if positions is not None:
            try:
                action = positions.index(position)
            except ValueError:
                # If exact match not found, find closest position
                action = min(range(len(positions)), 
                           key=lambda i: abs(positions[i] - position))
        else:
            action = position
        
        next_obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += float(reward)

        # Collect relevant information from the step
        price = _first_present(info, ("price", "close", "mid_price")) or _get_env_attr(env, ("price", "current_price"))
        portfolio_value = _first_present(info, ("portfolio_value", "net_worth", "equity", "balance")) \
                          or _get_env_attr(env, ("portfolio_value", "net_worth", "equity"))
        position_val = _first_present(info, ("position", "current_position")) or _get_env_attr(env, ("position",)) or position

        steps.append({
            'step': step_idx,
            'action': int(action) if isinstance(action, (int, np.integer)) else action,
            'position': _to_float(position_val),
            'reward': float(reward),
            'cumulative_reward': cumulative_reward,
            'price': _to_float(price),
            'portfolio_value': _to_float(portfolio_value),
            'timestamp': _first_present(info, ("timestamp", "datetime", "date", "time"))
        })
        obs = next_obs
        step_idx += 1
        if max_steps is not None and step_idx >= max_steps:
            break

    return {
        'steps': steps,
        'total_reward': cumulative_reward,
        'num_steps': len(steps)
    }

