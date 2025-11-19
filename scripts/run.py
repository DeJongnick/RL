"""Simple script to launch training or evaluation of a DQN model.

This script allows you to easily choose the configuration and data,
then launch training or evaluation of the model.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.train import train_agent, train_all_agents, load_config, resolve_project_path
from scripts.evaluate import evaluate_agent_model, evaluate_all_agents, evaluate_dqn_agent


def list_available_configs():
    """List available configurations."""
    configs_dir = PROJECT_ROOT / "configs"
    configs = []
    if configs_dir.exists():
        for config_file in sorted(configs_dir.glob("*.yaml")):
            configs.append(config_file.name)
    return configs


def list_available_data():
    """List available data files."""
    data_dir = PROJECT_ROOT / "data"
    data_files = []
    
    # Only list processed data files
    processed_dir = data_dir / "processed"
    if processed_dir.exists():
        for data_file in sorted(processed_dir.glob("*.csv")):
            data_files.append(f"processed/{data_file.name}")
    
    return data_files


def interactive_mode():
    """Interactive mode to choose config and data."""
    print("\n" + "="*60)
    print("Interactive Mode - Launch Configuration")
    print("="*60 + "\n")
    
    # Configuration choice
    configs = list_available_configs()
    if not configs:
        print("No configuration found in configs/")
        raise ValueError("No configuration files available. Please add a YAML file in configs/")
    
    print("Available configurations:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config}")
    
    choice = input(f"\nChoose a configuration (1-{len(configs)}): ").strip()
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(configs):
            config_path = f"configs/{configs[idx]}"
        else:
            raise ValueError(f"Invalid choice. Please select a number between 1 and {len(configs)}")
    except ValueError as e:
        if "Invalid choice" in str(e):
            raise
        raise ValueError(f"Invalid input. Please select a number between 1 and {len(configs)}")
    
    # Data choice (optional - can use data from config)
    print("\n" + "-"*60)
    data_files = list_available_data()
    if not data_files:
        print("No data files found in data/")
        data_path = None  # Use data from config
    else:
        print("Available data files:")
        for i, data_file in enumerate(data_files, 1):
            print(f"  {i}. {data_file}")
        
        choice = input(f"\nChoose data (1-{len(data_files)}, or press Enter to use config default): ").strip()
        if not choice:
            data_path = None  # Use data from config
        else:
            try:
                idx = int(choice)
                if 1 <= idx <= len(data_files):
                    data_path = f"data/{data_files[idx - 1]}"
                else:
                    raise ValueError(f"Invalid choice. Please select a number between 1 and {len(data_files)}")
            except ValueError as e:
                if "Invalid choice" in str(e):
                    raise
                raise ValueError(f"Invalid input. Please select a number between 1 and {len(data_files)}")
    
    # Mode choice
    print("\n" + "-"*60)
    print("Execution mode:")
    print("  1. Training (train)")
    print("  2. Evaluation (evaluate)")
    print("  3. Both (train then evaluate)")
    
    mode = input("\nChoose mode (1-3): ").strip()
    
    # Device is automatically selected
    device = None  # Auto-detect (cuda if available, else cpu)
    
    return config_path, data_path, mode, device


def run_training(config_path, data_path=None, device=None, all_agents=True):
    """Launch training for all agents."""
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    print(f"Configuration: {config_path}")
    if data_path:
        print(f"Data: {data_path}")
    print("="*60 + "\n")
    
    config = load_config(config_path)
    
    # Modify data path if specified
    if data_path:
        data_path_resolved = resolve_project_path(data_path)
        config['data']['raw_data_path'] = str(data_path_resolved)
        config['data']['processed_data_path'] = str(data_path_resolved)
        print(f"Using data: {data_path_resolved}")
    
    try:
        if all_agents:
            results, run_dir = train_all_agents(config, device=device)
            print(f"\n✓ Training completed successfully!")
            print(f"  Run directory: {run_dir}")
            return results, run_dir
        else:
            agent, env, run_dir = train_agent(config, device=device)
            print(f"\n✓ Training completed successfully!")
            print(f"  Run directory: {run_dir}")
            return agent, env, run_dir
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_evaluation(config_path, run_dir=None, model_path=None, data_path=None, device=None):
    """Launch evaluation for all agents."""
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    print(f"Configuration: {config_path}")
    if run_dir:
        print(f"Run directory: {run_dir}")
    if model_path:
        print(f"Model: {model_path}")
    if data_path:
        print(f"Data: {data_path}")
    print("="*60 + "\n")
    
    config = load_config(config_path)
    
    # Modify data path if specified
    if data_path:
        data_path_resolved = resolve_project_path(data_path)
        config['data']['raw_data_path'] = str(data_path_resolved)
        config['data']['processed_data_path'] = str(data_path_resolved)
    
    # Find run directory if not specified
    if not run_dir:
        # Search for most recent run directory
        logs_dir = PROJECT_ROOT / "logs" / "runs"
        if logs_dir.exists():
            runs = sorted(logs_dir.glob("run_*"), reverse=True)
            if runs:
                run_dir = runs[0]
                print(f"Using most recent run directory: {run_dir}")
    
    # If we have a run_dir, evaluate all agents
    if run_dir:
        run_dir = resolve_project_path(run_dir)
        try:
            results = evaluate_all_agents(config, str(run_dir), device=device, compare_baselines=True)
            print(f"\n✓ Evaluation completed successfully!")
            return results
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    elif model_path:
        # Fallback to single agent evaluation
        try:
            results = evaluate_agent_model(config, str(model_path), device=device, compare_baselines=True)
            print(f"\n✓ Evaluation completed successfully!")
            return results
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print("✗ No run directory or model path provided")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Launch training or evaluation of a DQN model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage examples:

  # Interactive mode
  python scripts/run.py

  # Training with specific config
  python scripts/run.py --train --config configs/hyperparameters_v1.yaml

  # Evaluation with specific model
  python scripts/run.py --evaluate --config configs/hyperparameters_v1.yaml --model logs/runs/run_XXX/checkpoints/best_model.pt

  # Training with custom data
  python scripts/run.py --train --config configs/hyperparameters_v1.yaml --data data/raw/binance-BTCUSDT-1h.pkl
        """
    )
    
    parser.add_argument('--train', action='store_true', help='Launch training (trains all agents by default)')
    parser.add_argument('--evaluate', action='store_true', help='Launch evaluation (evaluates all agents by default)')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data (optional)')
    parser.add_argument('--model', type=str, help='Path to model (for single agent evaluation)')
    parser.add_argument('--run-dir', type=str, help='Run directory (for multi-agent evaluation)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode if requested or if no arguments
    if args.interactive or (not args.train and not args.evaluate):
        config_path, data_path, mode, device = interactive_mode()
        
        if mode == '1' or mode.lower() == 'train':
            results, run_dir = run_training(config_path, data_path, device, all_agents=True)
            if results is not None and run_dir is not None:
                print(f"\n✓ All agents trained successfully!")
        elif mode == '2' or mode.lower() == 'evaluate':
            run_evaluation(config_path, run_dir=None, data_path=data_path, device=device)
        elif mode == '3' or mode.lower() == 'both':
            results, run_dir = run_training(config_path, data_path, device, all_agents=True)
            if results is not None and run_dir is not None:
                # Evaluate all agents in the run
                run_evaluation(config_path, run_dir=str(run_dir), data_path=data_path, device=device)
    else:
        # Command line mode
        device = torch.device(args.device) if args.device else None
        
        if args.train:
            run_training(args.config or 'configs/hyperparameters_v1.yaml', args.data, device, all_agents=True)
        
        if args.evaluate:
            run_evaluation(args.config or 'configs/hyperparameters_v1.yaml', 
                          run_dir=args.run_dir, model_path=args.model, data_path=args.data, device=device)


if __name__ == '__main__':
    main()
