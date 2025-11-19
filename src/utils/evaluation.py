"""Evaluation utilities for measuring agent performance."""

import pandas as pd
import time
from pathlib import Path


def evaluate_agent(
    agent, env, num_episodes=10, max_steps=None,
    render=True, csv_path="evaluation_results.csv",
    renderer_logs_dir="render_logs"
):
    """
    Evaluate an agent over multiple episodes of a trading environment.
    
    Args:
        agent: The agent to evaluate (must have act() method)
        env: The trading environment
        num_episodes: Number of episodes to run
        max_steps: Maximum steps per episode (None for no limit)
        render: Whether to save render logs
        csv_path: Path to save evaluation results CSV
        renderer_logs_dir: Directory to save render logs
    
    Returns:
        DataFrame with evaluation results
    """
    results = []
    if renderer_logs_dir:
        renderer_logs_dir = Path(renderer_logs_dir)
        renderer_logs_dir.mkdir(parents=True, exist_ok=True)

    # Get positions from environment or agent
    positions = getattr(env, 'positions', None) or getattr(agent, 'positions', None)
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        done, truncated = False, False
        step, total_reward = 0, 0.0

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
                # Assume env.step accepts position directly
                action = position
            
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            if max_steps and step >= max_steps:
                break

        # Retrieve and convert episode metrics
        metrics = env.get_metrics() if hasattr(env, 'get_metrics') else {}
        try:
            port_ret = float(metrics.get("Portfolio Return", "0%").rstrip('%')) / 100
            market_ret = float(metrics.get("Market Return", "0%").rstrip('%')) / 100
        except Exception:
            port_ret, market_ret = 0.0, 0.0

        results.append({
            "episode": ep + 1,
            "portfolio_return": port_ret,
            "market_return": market_ret,
            "excess_return": port_ret - market_ret,
            "steps": step,
            "total_reward": total_reward,
        })

        # Save visualizations/logs for each episode if requested
        if render and renderer_logs_dir and hasattr(env, 'save_for_render'):
            print(f"Ep {ep+1}: Reward={total_reward:.2f}, PortRet={port_ret:.2%}, MarketRet={market_ret:.2%}, Excess={port_ret-market_ret:.2%}, Steps={step}")
            time.sleep(1)
            env.save_for_render(dir=renderer_logs_dir)

    # Aggregate results as a DataFrame, then export to CSV
    df = pd.DataFrame(results)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    return df

