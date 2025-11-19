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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "eval"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent
from src.utils.data_loader import load_and_preprocess_data
from src.utils.environment_setup import create_eval_env
from src.utils.visualization import (
    plot_training_curves,
    plot_trading_decisions,
    plot_portfolio_performance,
    plot_trading_actions_timeline,
    plot_baseline_comparison,
    collect_episode_data
)


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
    """Plot training learning curves."""
    print("Training curves...")
    log_cfg = config.get('logging', {})
    log_dir = resolve_project_path(log_cfg.get('log_dir', 'logs'))
    csv_file = Path(log_cfg.get('csv_log_file', 'training_log.csv'))
    
    # If run_dir is provided, use it; otherwise use save_dir
    if run_dir:
        save_dir = Path(run_dir) / "eval" / "png"
    else:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = resolve_project_path(save_dir)
    
    # Try to find training log in run directory if run_dir is provided
    if run_dir:
        run_training_log = Path(run_dir) / "training_log.csv"
        if run_training_log.exists():
            csv_file = run_training_log

    candidate_paths = []
    if csv_file.is_absolute():
        candidate_paths.append(csv_file)
    else:
        if csv_file.parent == Path('.'):
            candidate_paths.append(log_dir / csv_file)
        candidate_paths.append(resolve_project_path(csv_file))
    if not candidate_paths:
        candidate_paths.append(log_dir / csv_file)

    csv_path = next((p for p in candidate_paths if p.exists()), candidate_paths[0])
    try:
        if not csv_path.exists():
            raise FileNotFoundError(f"No training log found ({csv_path})")
        save_path = Path(save_dir) / 'training_curves.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plot_training_curves(csv_path=csv_path, save_path=str(save_path))
        print(f"  ✓ {save_path}")
    except Exception as e:
        print(f"  ✗ {e}")


def generate_evaluation_plots(config, save_dir='eval', run_dir=None):
    """Plot evaluation results and compare with baselines."""
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
    
    # Baselines: CSVs must exist for comparison
    if run_dir:
        baseline_csvs = {
            'Random': csv_dir / 'random_results.csv',
            'DQN': csv_dir / 'dqn_results.csv'
        }
    else:
        baseline_csvs = {
            'Random': EVAL_ROOT / 'csv' / 'random_results.csv',
            'DQN': eval_csv
        }
    results = {k: pd.read_csv(p) for k, p in baseline_csvs.items() if p.exists()}
    if len(results) > 1:
        save_path = save_dir / 'baseline_comparison.png'
        plot_baseline_comparison(results, save_path=str(save_path))
        print(f"  ✓ Baseline comparison {save_path}")
    else:
        print("  ⚠ Missing baselines for comparison")


def generate_trading_visualizations(config, model_path, save_dir='eval', num_episodes=1, max_steps=None, device=None, run_dir=None):
    """Detailed visualization of trading episodes for trained agent."""
    print("Trading visualizations...")
    device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # If run_dir is provided, use it; otherwise use save_dir
    if run_dir:
        save_dir = Path(run_dir) / "eval" / "png"
    else:
        save_dir = Path(save_dir)
        if not save_dir.is_absolute():
            save_dir = resolve_project_path(save_dir)
    data_cfg = config['data']
    data_folder = resolve_project_path(data_cfg['data_folder'])
    _, _, df_eval = load_and_preprocess_data(
        data_folder=str(data_folder),
        download_if_missing=data_cfg.get('download_if_missing', True)
    )
    env_cfg = config['environment']
    env = create_eval_env(
        df_eval=df_eval,
        positions=env_cfg['positions'],
        trading_fees=env_cfg['trading_fees'],
        borrow_interest_rate=env_cfg['borrow_interest_rate']
    )
    obs, _ = env.reset()
    state_dim = len(obs)
    agent_cfg = config['agent']
    network_conf = config.get('network', {})
    model_path = resolve_project_path(model_path)
    if not model_path.exists():
        print(f"  ✗ DQN model checkpoint not found at {model_path}")
        return
    agent = DQNAgent(
        state_dim=state_dim,
        positions=env_cfg['positions'],
        config_path=None,
        lr=agent_cfg['lr'],
        gamma=agent_cfg['gamma'],
        epsilon_start=agent_cfg['epsilon_start'],
        epsilon_end=agent_cfg['epsilon_end'],
        epsilon_decay=agent_cfg['epsilon_decay'],
        buffer_size=agent_cfg['buffer_size'],
        batch_size=agent_cfg['batch_size'],
        target_update_freq=agent_cfg['target_update_freq'],
        hidden_dim=network_conf.get('hidden_dim', 128),
        device=device
    )
    agent.load(str(model_path))
    agent.q_network.eval()
    save_dir.mkdir(parents=True, exist_ok=True)
    for ep in range(num_episodes):
        print(f"  Episode {ep + 1}/{num_episodes}...")
        episode_data = collect_episode_data(env, agent, max_steps=max_steps)
        # 3 visualizations per episode (decisions, performance, action timeline):
        try:
            save_path = save_dir / f'trading_decisions_ep{ep+1}.png'
            plot_trading_decisions(env, agent, save_path=str(save_path), episode_data=episode_data)
            print(f"    ✓ {save_path}")
        except Exception as e:
            print(f"    ✗ trading_decisions : {e}")
        try:
            save_path = save_dir / f'portfolio_performance_ep{ep+1}.png'
            plot_portfolio_performance(episode_data, save_path=str(save_path))
            print(f"    ✓ {save_path}")
        except Exception as e:
            print(f"    ✗ portfolio_performance : {e}")
        try:
            save_path = save_dir / f'trading_timeline_ep{ep+1}.png'
            plot_trading_actions_timeline(episode_data, env, save_path=str(save_path))
            print(f"    ✓ {save_path}")
        except Exception as e:
            print(f"    ✗ trading_timeline : {e}")


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
        generate_trading_visualizations(config, args.model, save_dir=str(output_dir),
            num_episodes=args.num_episodes, max_steps=args.max_steps, device=device, run_dir=run_dir)
    else:
        print("=" * 60)
        print("Generating all visualizations")
        print("=" * 60)
        print("\n1. Training curves")
        generate_training_curves(config, save_dir=str(output_dir), run_dir=run_dir)
        print("\n2. Evaluation plots")
        generate_evaluation_plots(config, save_dir=str(output_dir), run_dir=run_dir)
        print("\n3. Detailed trading")
        generate_trading_visualizations(config, args.model, save_dir=str(output_dir),
            num_episodes=args.num_episodes, max_steps=args.max_steps, device=device, run_dir=run_dir)
        print("\n" + "=" * 60)
        if run_dir:
            print(f"All outputs saved in: {run_dir}/eval/png")
        else:
            print(f"All outputs saved in: {output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    main()

