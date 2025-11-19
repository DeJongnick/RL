# RL - Trading Agent with Deep Q-Learning

Reinforcement learning project for algorithmic trading using a DQN (Deep Q-Network) agent.

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Installation Check

```bash
python scripts/setup_check.py
```

## 📋 Usage

### Main Script: `run.py`

The `run.py` script allows you to easily launch training or evaluation of a model with choice of configuration and data.

#### Interactive Mode (Recommended)

```bash
python scripts/run.py
# or
python scripts/run.py --interactive
```

Interactive mode guides you through:
1. Configuration choice (among available YAML files)
2. Data choice (optional - uses config data by default)
3. Mode choice (training, evaluation, or both)
4. Device choice (cuda/cpu)

#### Command Line Mode

**Training:**
```bash
# With default configuration
python scripts/run.py --train

# With specific configuration
python scripts/run.py --train --config configs/hyperparameters_v1.yaml

# With custom data
python scripts/run.py --train --config configs/hyperparameters_v1.yaml --data data/raw/binance-BTCUSDT-1h.pkl

# With specific device
python scripts/run.py --train --config configs/hyperparameters_v1.yaml --device cuda
```

**Evaluation:**
```bash
# Evaluation with best model found automatically
python scripts/run.py --evaluate --config configs/hyperparameters_v1.yaml

# Evaluation with specific model
python scripts/run.py --evaluate --config configs/hyperparameters_v1.yaml --model logs/runs/run_XXX/checkpoints/best_model.pt
```

**Training then evaluation:**
```bash
python scripts/run.py --train --evaluate --config configs/hyperparameters_v1.yaml
```

### Individual Scripts

#### Training

```bash
python scripts/train.py --config configs/hyperparameters_v1.yaml
```

#### Evaluation

```bash
python scripts/evaluate.py --config configs/hyperparameters_v1.yaml --model logs/checkpoints/best_model.pt
```

#### Visualization

```bash
# All visualizations
python scripts/visualize.py --config configs/hyperparameters_v1.yaml --model logs/checkpoints/best_model.pt

# Only training curves
python scripts/visualize.py --training-only --config configs/hyperparameters_v1.yaml

# Only evaluation results
python scripts/visualize.py --evaluation-only --config configs/hyperparameters_v1.yaml
```

#### List Runs

```bash
# List all runs
python scripts/list_runs.py

# Details of a specific run
python scripts/list_runs.py --details run_20241119_153447

# Compare runs by metric
python scripts/list_runs.py --compare best_reward --top-n 10
```

## 📁 Project Structure

```
RL/
├── configs/              # YAML configuration files
│   ├── hyperparameters_v1.yaml
│   └── hyperparameters_v2.yaml
├── data/                 # Market data
│   ├── raw/              # Raw data
│   └── processed/        # Processed data
├── scripts/              # Execution scripts
│   ├── run.py           # Main script (interactive mode)
│   ├── train.py          # Training
│   ├── evaluate.py       # Evaluation
│   ├── visualize.py     # Visualization
│   ├── list_runs.py      # Run management
│   └── setup_check.py    # Installation check
├── src/
│   ├── agents/          # Agent implementations
│   │   ├── dqn_agent.py
│   │   └── random_agent.py
│   ├── models/          # Network architectures
│   │   └── dqn_network.py
│   └── utils/           # Utilities
│       ├── data_loader.py
│       ├── environment_setup.py
│       ├── evaluation.py
│       └── visualization.py
└── logs/                # Logs and results
    └── runs/            # Training runs (by timestamp)
```

## 🎯 Available Agents

- **DQNAgent**: Deep Q-Learning agent with Dueling DQN architecture
- **RandomAgent**: Baseline agent that selects random positions

## 📊 Configuration

YAML configuration files allow you to define:
- Agent hyperparameters (learning rate, epsilon, etc.)
- Network architecture
- Training parameters
- Environment configuration
- Data paths

See `configs/hyperparameters_v1.yaml` for a complete example.

## 📈 Results

Results are saved in `logs/runs/run_YYYYMMDD_HHMMSS/` with:
- `checkpoints/`: Saved models
- `training_log.csv`: Training logs
- `training_summary.json`: Training summary
- `config.yaml`: Used configuration
- `eval/`: Evaluation results (if evaluated)

## 🔧 Main Dependencies

- PyTorch
- Gymnasium
- gym-trading-env
- pandas
- numpy
- matplotlib
- seaborn
- tensorboard
- tqdm
- pyyaml

See `requirements.txt` for the complete list.
