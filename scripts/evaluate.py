"""Evaluation script for trained DQN trading agent."""

import argparse
import sys
from pathlib import Path

import torch

try:
    import yaml
except ImportError:
    print("Error: PyYAML is not installed. Please install it with: pip install pyyaml")
    print("Or install all requirements: pip install -r requirements.txt")
    exit(1)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent
from src.agents.random_agent import RandomAgent
from src.utils.data_loader import load_and_preprocess_data
from src.utils.environment_setup import create_eval_env
from src.utils.evaluation import evaluate_agent
from src.utils.visualization import plot_baseline_comparison


def resolve_project_path(path_like):
    """Resolve a path relative to project root if it's not absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path='configs/hyperparameters_v1.yaml'):
    """Load configuration YAML (relative path is accepted)."""
    config_path = resolve_project_path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


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


def evaluate_dqn_agent(config, model_path, device=None, compare_baselines=True, run_dir=None):
    """
    Evaluate the trained DQN agent. Optionally compare to baselines.
    
    Args:
        config: Configuration dictionary
        model_path: Path to the model checkpoint
        device: PyTorch device
        compare_baselines: Whether to evaluate baseline agents
        run_dir: Optional run directory to save results. If None, will try to detect from model_path.
    """
    model_path_resolved = resolve_project_path(model_path)
    
    # Try to detect run directory from model path if not provided
    if run_dir is None:
        run_dir = find_run_dir_from_model_path(model_path_resolved)
    
    # Set up output directories
    if run_dir:
        # Save in the run directory
        run_dir = Path(run_dir)
        eval_dir = run_dir / "eval"
        png_dir = eval_dir / "png"
        csv_dir = eval_dir / "csv"
        print(f"\nSaving evaluation results to run directory: {run_dir}")
    else:
        # Fallback to default eval directory
        eval_root = PROJECT_ROOT / "eval"
        png_dir = eval_root / "png"
        csv_dir = eval_root / "csv"
        print(f"\nSaving evaluation results to default directory: {eval_root}")
    
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    # PyTorch device
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data and set up evaluation environment (on test data)
    data_conf = config["data"]
    data_folder = resolve_project_path(data_conf["data_folder"])
    _, _, df_eval = load_and_preprocess_data(
        data_folder=str(data_folder),
        download_if_missing=data_conf.get('download_if_missing', True)
    )
    env_conf = config["environment"]
    env = create_eval_env(
        df_eval=df_eval,
        positions=env_conf['positions'],
        trading_fees=env_conf['trading_fees'],
        borrow_interest_rate=env_conf['borrow_interest_rate']
    )

    obs, _ = env.reset()
    state_dim = len(obs)

    # Load DQN agent (and weights)
    ag_conf = config["agent"]
    network_conf = config.get('network', {})
    model_path = resolve_project_path(model_path)
    
    agent = DQNAgent(
        state_dim=state_dim,
        positions=env_conf['positions'],
        config_path=None,
        lr=ag_conf['lr'],
        gamma=ag_conf['gamma'],
        epsilon_start=ag_conf['epsilon_start'],
        epsilon_end=ag_conf['epsilon_end'],
        epsilon_decay=ag_conf['epsilon_decay'],
        buffer_size=ag_conf['buffer_size'],
        batch_size=ag_conf['batch_size'],
        target_update_freq=ag_conf['target_update_freq'],
        hidden_dim=network_conf.get('hidden_dim', 128),
        device=device
    )
    agent.load(str(model_path))
    agent.epsilon = ag_conf['epsilon_end']  # Pure exploitation

    # --- DQN Evaluation ---
    print("\nEvaluating DQN agent...")
    eval_conf = config["evaluation"]
    dqn_csv = csv_dir / "dqn_results.csv"
    results_dqn = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=eval_conf['num_episodes'],
        render=eval_conf.get('render', False),
        csv_path=str(dqn_csv),
        renderer_logs_dir=str(eval_conf.get('render_logs_dir', 'eval/render_logs'))
    )

    print(f"\nDQN Results:\n"
          f"  Portfolio Return: {results_dqn['portfolio_return'].mean():.2%}\n"
          f"  Market Return   : {results_dqn['market_return'].mean():.2%}\n"
          f"  Excess Return   : {results_dqn['excess_return'].mean():.2%}")

    results_all = {"DQN": results_dqn}

    # --- Baselines ---
    if compare_baselines:
        print("\nEvaluating baselines...")
        # Random Agent
        random_agent = RandomAgent(positions=env_conf['positions'])
        random_csv = csv_dir / "random_results.csv"
        res_rand = evaluate_agent(random_agent, env, eval_conf['num_episodes'], 
                                  render=False, csv_path=str(random_csv))
        results_all["Random"] = res_rand
        print(f"Random Portfolio Return: {res_rand['portfolio_return'].mean():.2%}")

        # Visual comparison
        cmp_png = png_dir / "baseline_comparison.png"
        plot_baseline_comparison(results_all, save_path=str(cmp_png))
        print(f"Baseline comparison plot saved to: {cmp_png}")

    return results_all


def main():
    parser = argparse.ArgumentParser(description="Evaluate the trained DQN trading agent")
    parser.add_argument('--config', type=str, default='configs/hyperparameters_v1.yaml',
                        help='Path to configuration YAML file')
    parser.add_argument('--model', type=str, default='logs/checkpoints/best_model.pt',
                        help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default=None, help='cuda/cpu')
    parser.add_argument('--no-baselines', action='store_true', help="Don't evaluate baselines")
    parser.add_argument('--run-dir', type=str, default=None,
                        help='Run directory to save results (auto-detected from model path if not specified)')

    args = parser.parse_args()
    config = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    
    run_dir = resolve_project_path(args.run_dir) if args.run_dir else None

    evaluate_dqn_agent(config, args.model, device=device, compare_baselines=not args.no_baselines, run_dir=run_dir)
    print("\nEvaluation complete.")


if __name__ == '__main__':
    main()

