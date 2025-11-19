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
from scripts.train import train_agent, load_config, resolve_project_path
from scripts.evaluate import evaluate_dqn_agent


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


def run_training(config_path, data_path=None, device=None):
    """Launch training."""
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
        agent, env, run_dir = train_agent(config, device=device)
        print(f"\n✓ Training completed successfully!")
        print(f"  Run directory: {run_dir}")
        return agent, env, run_dir
    except Exception as e:
        print(f"\n✗ Error during training: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def run_evaluation(config_path, model_path=None, data_path=None, device=None):
    """Launch evaluation."""
    print("\n" + "="*60)
    print("STARTING EVALUATION")
    print("="*60)
    print(f"Configuration: {config_path}")
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
    
    # Find model if not specified
    if not model_path:
        # Search for best model in recent runs
        logs_dir = PROJECT_ROOT / "logs" / "runs"
        if logs_dir.exists():
            runs = sorted(logs_dir.glob("run_*"), reverse=True)
            for run_dir in runs:
                best_model = run_dir / "checkpoints" / "best_model.pt"
                if best_model.exists():
                    model_path = best_model
                    print(f"Model found: {model_path}")
                    break
        
        if not model_path:
            model_path = PROJECT_ROOT / "logs" / "checkpoints" / "best_model.pt"
            if not model_path.exists():
                model_path = input("No model found. Enter path to model: ").strip()
    
    try:
        results = evaluate_dqn_agent(config, str(model_path), device=device, compare_baselines=True)
        print(f"\n✓ Evaluation completed successfully!")
        return results
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
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
    
    parser.add_argument('--train', action='store_true', help='Launch training')
    parser.add_argument('--evaluate', action='store_true', help='Launch evaluation')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--data', type=str, help='Path to data (optional)')
    parser.add_argument('--model', type=str, help='Path to model (for evaluation)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], help='Device to use')
    parser.add_argument('--interactive', '-i', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Interactive mode if requested or if no arguments
    if args.interactive or (not args.train and not args.evaluate):
        config_path, data_path, mode, device = interactive_mode()
        
        if mode == '1' or mode.lower() == 'train':
            run_training(config_path, data_path, device)
        elif mode == '2' or mode.lower() == 'evaluate':
            run_evaluation(config_path, None, data_path, device)
        elif mode == '3' or mode.lower() == 'both':
            agent, env, run_dir = run_training(config_path, data_path, device)
            if agent is not None and run_dir is not None:
                # Find model in the run
                model_path = run_dir / "checkpoints" / "best_model.pt"
                if model_path.exists():
                    run_evaluation(config_path, str(model_path), data_path, device)
    else:
        # Command line mode
        device = torch.device(args.device) if args.device else None
        
        if args.train:
            run_training(args.config or 'configs/hyperparameters_v1.yaml', args.data, device)
        
        if args.evaluate:
            run_evaluation(args.config or 'configs/hyperparameters_v1.yaml', args.model, args.data, device)


if __name__ == '__main__':
    main()
