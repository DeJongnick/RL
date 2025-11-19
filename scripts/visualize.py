"""Synthetic visualization of the trading agent project results."""

import argparse
import sys
from pathlib import Path

import torch

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it with: pip install pyyaml")
    exit(1)

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import create_agent
from src.utils.data_loader import load_and_preprocess_data
from src.utils.environment_setup import create_eval_env
from src.utils.visualization import (
    plot_training_curves,
    plot_trading_decisions,
    plot_portfolio_performance,
    plot_trading_actions_timeline,
    plot_baseline_comparison,
    collect_episode_data,
    plot_comparative_trading_decisions,
    plot_comparative_portfolio_performance,
    plot_comparative_trading_timeline
)
import matplotlib.pyplot as plt


def resolve_project_path(path_like, base=None):
    """Resolve a path relative to project root if it's not absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    base_path = PROJECT_ROOT if base is None else base
    return base_path / path


def find_run_dir_from_model_path(model_path):
    """
    Try to find the run directory from the model path.
    If model is in logs/runs/run_XXX/checkpoints/, return the run directory.
    """
    model_path = Path(model_path).resolve()
    
    # Check if we're in a run directory structure
    parts = model_path.parts
    if 'runs' in parts:
        runs_idx = parts.index('runs')
        if runs_idx + 1 < len(parts):
            run_name = parts[runs_idx + 1]
            if run_name.startswith('run_'):
                # Reconstruct the run directory path
                run_dir = Path(*parts[:runs_idx+2])
                if run_dir.exists():
                    return run_dir
    
    return None


def load_config(config_path='configs/hyperparameters_v1.yaml'):
    """Load YAML config, automatically resolve relative paths."""
    config_path = resolve_project_path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)




def generate_training_curves(config, save_dir='eval', run_dir=None):
    """Plot training learning curves for all agents."""
    print("Training curves...")
    
    # If run_dir is provided, use it; otherwise use save_dir
    if run_dir:
        save_dir = Path(run_dir) / "eval" / "png"
        run_dir_path = Path(run_dir)
    else:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = resolve_project_path(save_dir)
        run_dir_path = None
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all training logs
    agent_types = ['dqn', 'ppo']
    training_logs = {}
    
    for agent_type in agent_types:
        if run_dir_path:
            csv_path = run_dir_path / f"training_log_{agent_type}.csv"
        else:
            log_cfg = config.get('logging', {})
            log_dir = resolve_project_path(log_cfg.get('log_dir', 'logs'))
            csv_path = log_dir / f"training_log_{agent_type}.csv"
        
        if csv_path.exists():
            training_logs[agent_type.upper()] = csv_path
    
    if not training_logs:
        print("  ⚠ No training logs found")
        return
    
    # Plot training curves for each agent
    for agent_name, csv_path in training_logs.items():
        try:
            save_path = save_dir / f'training_curves_{agent_name.lower()}.png'
            plot_training_curves(csv_path=csv_path, save_path=str(save_path))
            print(f"  ✓ {agent_name} training curves: {save_path}")
        except Exception as e:
            print(f"  ✗ Error plotting {agent_name} curves: {e}")
    
    # Create combined comparison plot if multiple agents
    if len(training_logs) > 1:
        try:
            save_path = save_dir / 'training_curves_comparison.png'
            plot_training_curves_comparison(training_logs, save_path=str(save_path))
            print(f"  ✓ Training curves comparison: {save_path}")
        except Exception as e:
            print(f"  ✗ Error creating comparison: {e}")


def plot_training_curves_comparison(training_logs, save_path=None):
    """
    Plot comparison of training curves for multiple agents.
    
    Args:
        training_logs: Dictionary mapping agent names to CSV paths
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for agent_name, csv_path in training_logs.items():
        df = pd.read_csv(csv_path)
        if 'episode' not in df:
            df['episode'] = np.arange(1, len(df) + 1)
        
        # Reward
        axes[0,0].plot(df['episode'], df.get('reward', pd.Series([0]*len(df))), 
                      label=agent_name, alpha=0.7)
        
        # Loss
        if 'loss' in df:
            axes[0,1].plot(df['episode'], df['loss'], label=agent_name, alpha=0.7)
        
        # Portfolio return
        if 'portfolio_return' in df:
            axes[1,0].plot(df['episode'], df['portfolio_return'] * 100, 
                          label=agent_name, alpha=0.7)
        
        # Excess return
        if 'excess_return' in df:
            axes[1,1].plot(df['episode'], df['excess_return'] * 100, 
                          label=agent_name, alpha=0.7)
    
    axes[0,0].set(title='Episode Reward Comparison', xlabel='Episode', ylabel='Reward')
    axes[0,1].set(title='Training Loss Comparison', xlabel='Episode', ylabel='Loss')
    axes[1,0].set(title='Portfolio Return Comparison', xlabel='Episode', ylabel='Return (%)')
    axes[1,1].set(title='Excess Return Comparison', xlabel='Episode', ylabel='Excess Return (%)')
    
    for ax in axes.flat:
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    plt.close()


def generate_evaluation_plots(config, save_dir='eval', run_dir=None):
    """Plot evaluation results and compare all agents."""
    print("Evaluation plots...")
    
    # If run_dir is provided, use it; otherwise use save_dir
    if run_dir:
        save_dir = Path(run_dir) / "eval" / "png"
        csv_dir = Path(run_dir) / "eval" / "csv"
    else:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = resolve_project_path(save_dir)
        csv_dir = EVAL_ROOT / "csv"
    
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all available agent results
    agent_types = ['dqn', 'ppo', 'random']
    results = {}
    
    for agent_type in agent_types:
        csv_path = csv_dir / f'{agent_type}_results.csv'
        if csv_path.exists():
            results[agent_type.upper()] = pd.read_csv(csv_path)
            print(f"  ✓ Loaded {agent_type.upper()} results")
    
    # Also check for agents_comparison.png (created by evaluate_all_agents)
    if run_dir:
        comparison_png = save_dir / 'agents_comparison.png'
        if comparison_png.exists():
            print(f"  ✓ Comparison plot already exists: {comparison_png}")
            return
    
    if len(results) > 1:
        save_path = save_dir / 'agents_comparison.png'
        plot_baseline_comparison(results, save_path=str(save_path))
        print(f"  ✓ Agents comparison saved to {save_path}")
    elif len(results) == 1:
        print(f"  ⚠ Only one agent result found, skipping comparison")
    else:
        print("  ⚠ No evaluation results found for comparison")


def generate_trading_visualizations(config, run_dir=None, num_episodes=1, max_steps=None, device=None):
    """Detailed visualization of trading episodes for all trained agents (aggregated comparison)."""
    print("Trading visualizations...")
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not run_dir:
        print("  ⚠ Run directory required for multi-agent visualizations")
        return
    
    run_dir = Path(run_dir)
    save_dir = run_dir / "eval" / "png"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    data_cfg = config['data']
    data_folder = resolve_project_path(data_cfg['data_folder'])
    _, _, df_eval = load_and_preprocess_data(
        data_folder=str(data_folder),
        download_if_missing=data_cfg.get('download_if_missing', True)
    )
    env_cfg = config['environment']
    agent_cfg = config['agent']
    network_conf = config.get('network', {})
    
    # Create environment (will be reused for each agent)
    env = create_eval_env(
        df_eval=df_eval,
        positions=env_cfg['positions'],
        trading_fees=env_cfg['trading_fees'],
        borrow_interest_rate=env_cfg['borrow_interest_rate']
    )
    obs, _ = env.reset()
    state_dim = len(obs)
    
    # Collect all agents and their models
    agents = {}
    
    # Import required modules
    from src.agents.random_agent import RandomAgent
    
    # Load DQN agent
    dqn_checkpoint_dir = run_dir / "checkpoints" / "dqn"
    dqn_best_model_path = dqn_checkpoint_dir / "best_model.pt"
    if dqn_best_model_path.exists():
        try:
            agent = create_agent(
                agent_type='dqn',
                state_dim=state_dim,
                env_conf=env_cfg,
                agent_conf=agent_cfg,
                network_conf=network_conf,
                device=device
            )
            agent.load(str(dqn_best_model_path))
            agent.network.eval()
            agents['DQN'] = agent
            print(f"  ✓ Loaded DQN agent")
        except Exception as e:
            print(f"  ✗ Error loading DQN agent: {e}")
    
    # Load PPO agent
    ppo_checkpoint_dir = run_dir / "checkpoints" / "ppo"
    ppo_best_model_path = ppo_checkpoint_dir / "best_model.pt"
    if ppo_best_model_path.exists():
        try:
            agent = create_agent(
                agent_type='ppo',
                state_dim=state_dim,
                env_conf=env_cfg,
                agent_conf=agent_cfg,
                network_conf=network_conf,
                device=device
            )
            agent.load(str(ppo_best_model_path))
            agent.network.eval()
            agents['PPO'] = agent
            print(f"  ✓ Loaded PPO agent")
        except Exception as e:
            print(f"  ✗ Error loading PPO agent: {e}")
    
    # Create Random agent
    try:
        random_agent = RandomAgent(positions=env_cfg['positions'])
        agents['Random'] = random_agent
        print(f"  ✓ Loaded Random agent")
    except Exception as e:
        print(f"  ✗ Error creating Random agent: {e}")
    
    if not agents:
        print("  ⚠ No agents available for visualization")
        return
    
    print(f"\n  Visualizing {len(agents)} agents: {', '.join(agents.keys())}")
    
    # Generate comparative visualizations for each episode
    for ep in range(num_episodes):
        print(f"\n  Episode {ep + 1}/{num_episodes}...")
        
        # Collect episode data for all agents
        agents_episode_data = {}
        for agent_name, agent in agents.items():
            try:
                # Reset environment for each agent
                obs, _ = env.reset()
                episode_data = collect_episode_data(env, agent, max_steps=max_steps)
                agents_episode_data[agent_name] = episode_data
                print(f"    ✓ Collected data for {agent_name}")
            except Exception as e:
                print(f"    ✗ Error collecting data for {agent_name}: {e}")
        
        if not agents_episode_data:
            print(f"    ⚠ No episode data collected for episode {ep + 1}")
            continue
        
        # Create comparative visualizations
        try:
            save_path = save_dir / f'aggregated_trading_decisions_ep{ep+1}.png'
            plot_comparative_trading_decisions(agents_episode_data, save_path=str(save_path))
            print(f"      ✓ Comparative trading decisions: {save_path}")
        except Exception as e:
            print(f"      ✗ Comparative trading decisions: {e}")
        
        try:
            save_path = save_dir / f'aggregated_portfolio_performance_ep{ep+1}.png'
            plot_comparative_portfolio_performance(agents_episode_data, save_path=str(save_path))
            print(f"      ✓ Comparative portfolio performance: {save_path}")
        except Exception as e:
            print(f"      ✗ Comparative portfolio performance: {e}")
        
        try:
            save_path = save_dir / f'aggregated_trading_timeline_ep{ep+1}.png'
            plot_comparative_trading_timeline(agents_episode_data, save_path=str(save_path))
            print(f"      ✓ Comparative trading timeline: {save_path}")
        except Exception as e:
            print(f"      ✗ Comparative trading timeline: {e}")
        
        # Also create individual plots for each agent (optional, for detailed analysis)
        for agent_name, episode_data in agents_episode_data.items():
            agent_type = agent_name.lower()
            try:
                save_path = save_dir / f'{agent_type}_trading_decisions_ep{ep+1}.png'
                plot_trading_decisions(env, agents[agent_name], save_path=str(save_path), episode_data=episode_data)
            except Exception as e:
                pass  # Skip individual plots if they fail
            
            try:
                save_path = save_dir / f'{agent_type}_portfolio_performance_ep{ep+1}.png'
                plot_portfolio_performance(episode_data, save_path=str(save_path))
            except Exception as e:
                pass  # Skip individual plots if they fail


def main():
    parser = argparse.ArgumentParser(description='Trading agent visualization plots')
    parser.add_argument('--config', type=str, default='configs/hyperparameters_v1.yaml')
    parser.add_argument('--model', type=str, default='logs/checkpoints/best_model.pt')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (auto-detected from model path if not specified)')
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Run directory to save results (auto-detected from model path if not specified)')
    parser.add_argument('--training-only', action='store_true')
    parser.add_argument('--evaluation-only', action='store_true')
    parser.add_argument('--trading-only', action='store_true')
    parser.add_argument('--num-episodes', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    device = args.device or None
    
    # Try to detect run directory from model path
    model_path_resolved = resolve_project_path(args.model)
    if args.run_dir:
        run_dir = resolve_project_path(args.run_dir)
    else:
        run_dir = find_run_dir_from_model_path(model_path_resolved)
    
    # Set output directory
    if run_dir:
        output_dir = Path(run_dir) / "eval" / "png"
        print(f"\nSaving visualizations to run directory: {run_dir}")
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = resolve_project_path(output_dir)
    else:
        output_dir = EVAL_ROOT / "png"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select which plots to generate
    if args.training_only:
        generate_training_curves(config, save_dir=str(output_dir), run_dir=run_dir)
    elif args.evaluation_only:
        generate_evaluation_plots(config, save_dir=str(output_dir), run_dir=run_dir)
    elif args.trading_only:
        generate_trading_visualizations(config, run_dir=str(run_dir) if run_dir else None,
            num_episodes=args.num_episodes, max_steps=args.max_steps, device=device)
    else:
        print("=" * 60)
        print("Generating all visualizations")
        print("=" * 60)
        print("\n1. Training curves")
        generate_training_curves(config, save_dir=str(output_dir), run_dir=run_dir)
        print("\n2. Evaluation plots")
        generate_evaluation_plots(config, save_dir=str(output_dir), run_dir=run_dir)
        print("\n3. Detailed trading")
        generate_trading_visualizations(config, run_dir=str(run_dir) if run_dir else None,
            num_episodes=args.num_episodes, max_steps=args.max_steps, device=device)
        print("\n" + "=" * 60)
        if run_dir:
            print(f"All outputs saved in: {run_dir}/eval/png")
        else:
            print(f"All outputs saved in: {output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()

