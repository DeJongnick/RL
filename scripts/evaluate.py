"""Evaluation script for trained trading agents (DQN, PPO, etc.)."""

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
from src.agents.ppo_agent import PPOAgent
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


def create_agent_for_eval(agent_type, state_dim, env_conf, agent_conf, network_conf, device):
    """
    Create an agent for evaluation based on the agent type.
    
    Args:
        agent_type: Type of agent ('dqn' or 'ppo')
        state_dim: Dimension of the state space
        env_conf: Environment configuration
        agent_conf: Agent configuration
        network_conf: Network configuration
        device: PyTorch device
        
    Returns:
        Initialized agent
    """
    agent_type = agent_type.lower()
    
    if agent_type == 'dqn':
        return DQNAgent(
            state_dim=state_dim,
            positions=env_conf['positions'],
            config_path=None,
            lr=agent_conf.get('lr', 0.0001),
            gamma=agent_conf.get('gamma', 0.99),
            epsilon_start=agent_conf.get('epsilon_start', 1.0),
            epsilon_end=agent_conf.get('epsilon_end', 0.01),
            epsilon_decay=agent_conf.get('epsilon_decay', 0.995),
            buffer_size=agent_conf.get('buffer_size', 100000),
            batch_size=agent_conf.get('batch_size', 64),
            target_update_freq=agent_conf.get('target_update_freq', 100),
            hidden_dim=network_conf.get('hidden_dim', 128),
            device=device
        )
    elif agent_type == 'ppo':
        return PPOAgent(
            state_dim=state_dim,
            positions=env_conf['positions'],
            config_path=None,
            lr=agent_conf.get('lr', 3e-4),
            gamma=agent_conf.get('gamma', 0.99),
            gae_lambda=agent_conf.get('gae_lambda', 0.95),
            clip_epsilon=agent_conf.get('clip_epsilon', 0.2),
            value_coef=agent_conf.get('value_coef', 0.5),
            entropy_coef=agent_conf.get('entropy_coef', 0.01),
            max_grad_norm=agent_conf.get('max_grad_norm', 0.5),
            update_epochs=agent_conf.get('update_epochs', 4),
            batch_size=agent_conf.get('batch_size', 64),
            hidden_dim=network_conf.get('hidden_dim', 128),
            device=device
        )
    else:
        raise ValueError(f"Unknown agent type: {agent_type}. Supported types: 'dqn', 'ppo'")


def evaluate_all_agents(config, run_dir, device=None, compare_baselines=True):
    """
    Evaluate all trained agents in a run directory and create comparisons.
    
    Args:
        config: Configuration dictionary
        run_dir: Run directory containing checkpoints for all agents
        device: PyTorch device
        compare_baselines: Whether to evaluate baseline agents
    
    Returns:
        Dictionary mapping agent names to result DataFrames
    """
    run_dir = Path(run_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set up output directories
    eval_dir = run_dir / "eval"
    png_dir = eval_dir / "png"
    csv_dir = eval_dir / "csv"
    png_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nEvaluating all agents in run directory: {run_dir}")
    
    # Load data and set up evaluation environment
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
    
    eval_conf = config["evaluation"]
    results_all = {}
    
    # Evaluate each agent type in specific order: PPO, DQN
    agent_types = ['ppo', 'dqn']  # Order: PPO first, then DQN
    ag_conf = config["agent"]
    network_conf = config.get('network', {})
    
    for agent_type in agent_types:
        checkpoint_dir = run_dir / "checkpoints" / agent_type
        best_model_path = checkpoint_dir / "best_model.pt"
        
        if not best_model_path.exists():
            print(f"\n⚠ No model found for {agent_type.upper()} at {best_model_path}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Evaluating {agent_type.upper()} agent...")
        print(f"{'='*60}")
        
        try:
            agent = create_agent_for_eval(
                agent_type=agent_type,
                state_dim=state_dim,
                env_conf=env_conf,
                agent_conf=ag_conf,
                network_conf=network_conf,
                device=device
            )
            agent.load(str(best_model_path))
            
            # Set to evaluation mode
            if agent_type == 'dqn':
                agent.epsilon = ag_conf.get('epsilon_end', 0.01)
                agent.q_network.eval()
                agent.target_network.eval()
            else:
                agent.network.eval()
            
            # Evaluate agent
            agent_csv = csv_dir / f"{agent_type}_results.csv"
            results_agent = evaluate_agent(
                agent=agent,
                env=env,
                num_episodes=eval_conf['num_episodes'],
                render=eval_conf.get('render', False),
                csv_path=str(agent_csv),
                renderer_logs_dir=str(eval_conf.get('render_logs_dir', 'eval/render_logs'))
            )
            
            results_all[agent_type.upper()] = results_agent
            
            print(f"\n{agent_type.upper()} Results:")
            print(f"  Portfolio Return: {results_agent['portfolio_return'].mean():.2%}")
            print(f"  Market Return   : {results_agent['market_return'].mean():.2%}")
            print(f"  Excess Return   : {results_agent['excess_return'].mean():.2%}")
            
        except Exception as e:
            print(f"\n✗ Error evaluating {agent_type.upper()}: {e}")
            import traceback
            traceback.print_exc()
    
    # Evaluate baselines if requested (Random will be added last)
    if compare_baselines:
        print(f"\n{'='*60}")
        print("Evaluating baselines...")
        print(f"{'='*60}")
        
        # Random Agent
        random_agent = RandomAgent(positions=env_conf['positions'])
        random_csv = csv_dir / "random_results.csv"
        res_rand = evaluate_agent(
            random_agent, env, eval_conf['num_episodes'], 
            render=False, csv_path=str(random_csv)
        )
        results_all["Random"] = res_rand
        print(f"Random Portfolio Return: {res_rand['portfolio_return'].mean():.2%}")
    
    # Create comparison plot if we have multiple agents
    # Ensure order: PPO, DQN, Random
    if len(results_all) > 1:
        print(f"\n{'='*60}")
        print("Creating comparison plot...")
        print(f"{'='*60}")
        
        # Reorder results to ensure PPO, DQN, Random order
        ordered_results = {}
        for agent_name in ['PPO', 'DQN', 'Random']:
            if agent_name in results_all:
                ordered_results[agent_name] = results_all[agent_name]
        
        # Add any other agents that might exist
        for agent_name, results in results_all.items():
            if agent_name not in ordered_results:
                ordered_results[agent_name] = results
        
        cmp_png = png_dir / "agents_comparison.png"
        plot_baseline_comparison(ordered_results, save_path=str(cmp_png))
        print(f"✓ Comparison plot saved to: {cmp_png}")
        print(f"  Agents compared: {', '.join(ordered_results.keys())}")
    
    return results_all


def evaluate_agent_model(config, model_path, device=None, compare_baselines=True, run_dir=None):
    """
    Evaluate the trained agent. Optionally compare to baselines.
    
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

    # Get agent type from config (default to 'dqn' for backward compatibility)
    agent_type = config.get('agent_type', 'dqn').lower()
    print(f"Agent type: {agent_type.upper()}")

    # Load agent (and weights)
    ag_conf = config["agent"]
    network_conf = config.get('network', {})
    model_path = resolve_project_path(model_path)
    
    agent = create_agent_for_eval(
        agent_type=agent_type,
        state_dim=state_dim,
        env_conf=env_conf,
        agent_conf=ag_conf,
        network_conf=network_conf,
        device=device
    )
    agent.load(str(model_path))
    
    # Set to evaluation mode (no exploration)
    if agent_type == 'dqn':
        agent.epsilon = ag_conf.get('epsilon_end', 0.01)  # Pure exploitation
        agent.q_network.eval()
        agent.target_network.eval()
    else:
        agent.network.eval()  # Set network to eval mode

    # --- Agent Evaluation ---
    agent_name = agent_type.upper()
    print(f"\nEvaluating {agent_name} agent...")
    eval_conf = config["evaluation"]
    agent_csv = csv_dir / f"{agent_type}_results.csv"
    results_agent = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=eval_conf['num_episodes'],
        render=eval_conf.get('render', False),
        csv_path=str(agent_csv),
        renderer_logs_dir=str(eval_conf.get('render_logs_dir', 'eval/render_logs'))
    )

    print(f"\n{agent_name} Results:\n"
          f"  Portfolio Return: {results_agent['portfolio_return'].mean():.2%}\n"
          f"  Market Return   : {results_agent['market_return'].mean():.2%}\n"
          f"  Excess Return   : {results_agent['excess_return'].mean():.2%}")

    results_all = {agent_name: results_agent}

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
    parser = argparse.ArgumentParser(description="Evaluate the trained trading agent")
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

    evaluate_agent_model(config, args.model, device=device, compare_baselines=not args.no_baselines, run_dir=run_dir)
    print("\nEvaluation complete.")


# Backward compatibility alias
evaluate_dqn_agent = evaluate_agent_model


if __name__ == '__main__':
    main()

