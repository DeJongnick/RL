# RL - Trading Agent with Deep Reinforcement Learning

A reinforcement learning project for algorithmic trading using Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) agents. This project implements state-of-the-art RL algorithms to learn optimal trading strategies in cryptocurrency markets.

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
│   │   ├── ppo_agent.py
│   │   └── random_agent.py
│   ├── models/          # Network architectures
│   │   ├── dqn_network.py
│   │   └── ppo_network.py
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
- **PPOAgent**: Proximal Policy Optimization agent with Actor-Critic architecture
- **RandomAgent**: Baseline agent that selects random positions

## 🧠 Conceptual Overview

### Problem Formulation

This project frames algorithmic trading as a **Markov Decision Process (MDP)**:

- **State Space**: 7-dimensional vector containing:
  - 5 market features (price, volume, technical indicators)
  - 2 position features (current position, portfolio value)
  
- **Action Space**: Discrete set of position sizes:
  - `-1`: Short position (betting against the market)
  - `0`: No position (hold cash)
  - `0.5`: Half long position (conservative long)
  - `1`: Full long position (standard long)
  - `2`: 2x leverage long position (aggressive long)

- **Reward Signal**: Portfolio return adjusted for:
  - Trading fees (0.01% per transaction)
  - Borrow interest (for short positions: 0.0003% per timestep)
  - Risk-adjusted returns

- **Objective**: Maximize cumulative discounted reward (portfolio value) over time

### Why Reinforcement Learning?

Traditional trading strategies rely on:
- **Rule-based systems**: Hard to adapt to changing market conditions
- **Supervised learning**: Requires labeled data (what is the "correct" trade?)
- **Time series forecasting**: Doesn't account for sequential decision-making

RL addresses these limitations by:
1. **Learning from interaction**: Agent learns optimal policy through trial and error
2. **Sequential decision-making**: Considers long-term consequences of actions
3. **Adaptive exploration**: Balances exploitation of known strategies with exploration of new ones
4. **No labeled data needed**: Only requires reward signal (portfolio performance)

## 🏗️ Architecture Deep Dive

### DQN Agent: Dueling Deep Q-Network

The DQN agent uses a **Dueling Architecture** that separates state value estimation from action advantage estimation. This design choice is crucial for trading because:

1. **State Value (V(s))**: Estimates how good it is to be in a given market state
2. **Action Advantage (A(s,a))**: Estimates how much better/worse a specific position is compared to the average

The final Q-value combines both:
```
Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
```

**Network Architecture:**

```
Input State (7 dim)
    │
    ├─→ FC(128) ──→ ReLU
    │
    ├─→ FC(128) ──→ ReLU
    │
    ├─→ ┌─────────────────────┐
    │   │   Value Stream       │
    │   │   FC(64) → ReLU      │
    │   │   FC(1) → V(s)       │
    │   └─────────────────────┘
    │
    └─→ ┌─────────────────────┐
        │   Advantage Stream   │
        │   FC(64) → ReLU      │
        │   FC(5) → A(s,a)     │
        └─────────────────────┘
                │
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
                │
        Output: Q-values for 5 actions
```

**Key Components:**

1. **Experience Replay Buffer** (100,000 capacity):
   - Stores past transitions (state, action, reward, next_state, done)
   - Random sampling breaks correlation between consecutive experiences
   - Enables learning from rare but important events

2. **Target Network**:
   - Separate network with frozen weights
   - Updated every 100 training steps
   - Provides stable targets for Q-learning updates
   - Prevents "chasing moving targets" problem

3. **Double DQN**:
   - Uses main network to select best action
   - Uses target network to evaluate that action
   - Reduces overestimation bias in Q-values

4. **Epsilon-Greedy Exploration**:
   - Starts at ε=1.0 (100% random exploration)
   - Decays to ε=0.01 (1% exploration) over training
   - Decay factor: 0.995 per episode
   - Balances exploration vs exploitation

### PPO Agent: Actor-Critic Architecture

The PPO agent uses an **Actor-Critic** architecture that directly learns a policy (probability distribution over actions) rather than Q-values.

**Network Architecture:**

```
Input State (7 dim)
    │
    ├─→ FC(128) ──→ ReLU
    │
    ├─→ FC(128) ──→ ReLU
    │
    ├─→ ┌─────────────────────┐
    │   │   Actor Head         │
    │   │   FC(5) → Logits     │
    │   │   Softmax → π(a|s)   │
    │   └─────────────────────┘
    │
    └─→ ┌─────────────────────┐
        │   Critic Head        │
        │   FC(1) → V(s)       │
        └─────────────────────┘
```

**Key Components:**

1. **Generalized Advantage Estimation (GAE)**:
   - Combines TD(λ) with variance reduction
   - λ=0.95 balances bias-variance tradeoff
   - Provides better advantage estimates than simple TD errors

2. **Clipped Surrogate Objective**:
   - Prevents policy from updating too aggressively
   - Clip parameter ε=0.2 limits policy change to ±20%
   - Ensures stable learning without catastrophic forgetting

3. **Multiple Update Epochs**:
   - Updates policy 4 times on same batch of data
   - Maximizes sample efficiency
   - Shuffles data between epochs to prevent overfitting

4. **Entropy Bonus**:
   - Coefficient: 0.01
   - Encourages exploration by maintaining policy diversity
   - Prevents premature convergence to deterministic policy

## ⚙️ Hyperparameter Rationale

### DQN Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 0.0001 | Conservative rate prevents Q-value instability. Trading rewards are noisy, so smaller steps are safer. |
| **Gamma (Discount)** | 0.99 | High discount factor (99%) because trading decisions have long-term consequences. Future portfolio value matters. |
| **Epsilon Start** | 1.0 | Full exploration initially. Market dynamics are complex, need to explore all position types. |
| **Epsilon End** | 0.01 | Maintains 1% exploration even after training. Markets change, need to adapt. |
| **Epsilon Decay** | 0.995 | Slow decay (0.5% per episode). With 500 episodes, reaches ~8% by end. Gradual transition to exploitation. |
| **Buffer Size** | 100,000 | Large buffer stores ~1 year of hourly data. Enables learning from diverse market conditions. |
| **Batch Size** | 64 | Standard size balances gradient variance and computational efficiency. |
| **Target Update Freq** | 100 | Updates target network every 100 steps. Frequent enough for stability, not too frequent to slow learning. |
| **Hidden Dim** | 128 | Moderate capacity. Trading state space is relatively small (7D), but needs capacity for non-linear patterns. |

### PPO Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Learning Rate** | 3e-4 | Standard PPO learning rate. Higher than DQN because policy gradient is more stable than Q-learning. |
| **Gamma** | 0.99 | Same as DQN - long-term thinking is crucial for trading. |
| **GAE Lambda** | 0.95 | High λ reduces variance in advantage estimates. Important for noisy trading rewards. |
| **Clip Epsilon** | 0.2 | Standard PPO clipping. Prevents policy from changing too much per update. |
| **Value Coef** | 0.5 | Balances policy and value learning. Value function helps reduce variance in policy updates. |
| **Entropy Coef** | 0.01 | Small bonus maintains exploration. Too high would prevent convergence, too low would overfit. |
| **Max Grad Norm** | 0.5 | Gradient clipping prevents exploding gradients. More aggressive than DQN (1.0) because PPO updates more frequently. |
| **Update Epochs** | 4 | Multiple passes over same data improve sample efficiency. 4 is standard for PPO. |
| **Batch Size** | 64 | Same as DQN for consistency. |

### Environment Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Trading Fees** | 0.0001 (0.01%) | Realistic fee for cryptocurrency exchanges. High enough to matter, low enough to allow profitable trading. |
| **Borrow Interest** | 0.000003 (0.0003%) | Interest rate for short positions per timestep. At hourly frequency, this compounds to realistic annual rates. |
| **Max Steps** | 8760 | One year of hourly data. Long enough to see market cycles, short enough for training. |

## 📊 Benchmark Results

### Training Performance (500 Episodes)

**DQN Agent:**
- **Best Episode**: Episode 362
- **Best Portfolio Return**: +53.49%
- **Best Excess Return**: +52.69% (vs market)
- **Average Portfolio Return**: +1.83%
- **Average Excess Return**: +1.04%
- **Final Portfolio Return**: +11.82%

**PPO Agent:**
- **Best Episode**: Episode 170
- **Best Portfolio Return**: +3.28%
- **Best Excess Return**: +2.49% (vs market)
- **Average Portfolio Return**: +0.25%
- **Average Excess Return**: -0.55%
- **Final Portfolio Return**: +0.40%

### Evaluation Performance (10 Episodes, Out-of-Sample)

**DQN Agent:**
- **Average Portfolio Return**: +4.28%
- **Average Market Return**: -3.63%
- **Average Excess Return**: +7.91%
- **Consistency**: Very stable (std: 0.01%)

**PPO Agent:**
- **Average Portfolio Return**: -0.15%
- **Average Market Return**: -3.63%
- **Average Excess Return**: +3.48%
- **Consistency**: High variance (std: 18.5%)

**Random Baseline:**
- **Average Portfolio Return**: ~0% (random walk)
- **Average Excess Return**: ~-3.63% (underperforms market)

### Key Insights

1. **DQN Outperforms PPO**: DQN's value-based approach is more stable for trading. The dueling architecture effectively separates market state value from position-specific advantages.

2. **DQN Consistency**: DQN shows remarkable consistency in evaluation (low variance), suggesting it learned a robust policy rather than overfitting to training data.

3. **PPO Variance**: PPO's higher variance indicates it may need more training or different hyperparameters. Policy gradient methods can be more sensitive to reward shaping.

4. **Market Outperformance**: Both agents significantly outperform the market (-3.63% return) and random baseline, demonstrating learned trading strategies.

5. **Best vs Average**: Large gap between best and average performance suggests:
   - Market conditions vary significantly
   - Agents may benefit from regime detection
   - Ensemble methods could improve robustness

## 🔬 Why These Architectures Work

### Dueling DQN for Trading

The dueling architecture is particularly well-suited for trading because:

1. **State Value Separation**: Market conditions (bull/bear/sideways) have intrinsic value independent of position. The value stream learns this.

2. **Action Advantage**: Different positions have different advantages in the same market state. The advantage stream captures position-specific value.

3. **Sample Efficiency**: By decomposing Q(s,a), the network can generalize better. If it learns that a market state is generally bad (low V(s)), it doesn't need to learn that every position is bad individually.

### Actor-Critic for Trading

The actor-critic architecture offers:

1. **Direct Policy Learning**: Learns probability distribution over positions, allowing for stochastic policies that can be more robust.

2. **Value Function**: The critic provides baseline for variance reduction, crucial for noisy trading rewards.

3. **On-Policy Learning**: PPO learns from current policy's experiences, which can be more stable in non-stationary markets.

## 📈 Training Dynamics

### DQN Training Curve Characteristics

1. **Initial Phase (Episodes 0-100)**:
   - High exploration (ε ≈ 1.0)
   - Random actions, negative returns
   - Network learning basic patterns

2. **Learning Phase (Episodes 100-300)**:
   - Decreasing exploration (ε ≈ 0.6 → 0.3)
   - Rapid improvement in returns
   - Q-values stabilizing

3. **Refinement Phase (Episodes 300-500)**:
   - Low exploration (ε ≈ 0.1 → 0.01)
   - Fine-tuning of policy
   - Occasional best episodes as agent explores remaining uncertainty

### PPO Training Curve Characteristics

1. **Initial Phase**: Faster initial learning due to direct policy optimization
2. **Mid Training**: More variance as policy explores different strategies
3. **Convergence**: Slower convergence than DQN, but potentially more stable long-term

## 🎓 Learning Algorithm Details

### DQN Update Rule

The DQN minimizes the Temporal Difference (TD) error:

```
Loss = (Q(s,a) - (r + γ * Q_target(s', a*)))^2
```

Where:
- `Q(s,a)`: Current Q-value estimate
- `r`: Immediate reward
- `γ`: Discount factor (0.99)
- `Q_target(s', a*)`: Target Q-value using Double DQN:
  - `a* = argmax_a Q_main(s')` (action selection)
  - `Q_target(s', a*)` (action evaluation)

### PPO Update Rule

PPO maximizes the clipped surrogate objective:

```
L^CLIP = E[min(
    ratio * A(s,a),
    clip(ratio, 1-ε, 1+ε) * A(s,a)
)]
```

Where:
- `ratio = π_new(a|s) / π_old(a|s)`: Policy change ratio
- `A(s,a)`: Advantage estimate (from GAE)
- `ε = 0.2`: Clipping parameter

The total loss includes:
- Policy loss (clipped objective)
- Value loss: `(V(s) - Returns)^2`
- Entropy bonus: `-H(π(a|s))`

## 🔧 Configuration

YAML configuration files allow you to define:
- Agent hyperparameters (learning rate, epsilon, etc.)
- Network architecture
- Training parameters
- Environment configuration
- Data paths

See `configs/hyperparameters_v1.yaml` for a complete example.

## 📈 Results Storage

Results are saved in `logs/runs/run_YYYYMMDD_HHMMSS/` with:
- `checkpoints/`: Saved models (best, final, periodic checkpoints)
- `training_log.csv`: Episode-by-episode training metrics
- `training_summary.json`: Aggregated training statistics
- `config.yaml`: Used configuration (for reproducibility)
- `eval/`: Evaluation results (CSV files and comparison plots)
- `tensorboard/`: TensorBoard logs for visualization

## 🔧 Main Dependencies

- **PyTorch**: Deep learning framework
- **Gymnasium**: RL environment interface
- **gym-trading-env**: Custom trading environment
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization
- **tensorboard**: Training monitoring
- **tqdm**: Progress bars
- **pyyaml**: Configuration management

See `requirements.txt` for the complete list.

## 🚀 Next Steps & Improvements

### Potential Enhancements

1. **Multi-Asset Trading**: Extend to portfolio of cryptocurrencies
2. **Regime Detection**: Add market regime classification (trending/volatile/sideways)
3. **Risk Management**: Incorporate position sizing based on volatility
4. **Ensemble Methods**: Combine multiple agents for robustness
5. **Attention Mechanisms**: Add transformer layers for long-term dependencies
6. **Hierarchical RL**: Separate high-level strategy from low-level execution
7. **Transfer Learning**: Pre-train on multiple assets, fine-tune on target

### Research Directions

- **Reward Shaping**: Experiment with risk-adjusted rewards (Sharpe ratio, Sortino ratio)
- **State Representation**: Add more technical indicators or use raw OHLCV with CNNs
- **Action Space**: Continuous actions for precise position sizing
- **Multi-Task Learning**: Learn to trade multiple timeframes simultaneously
---

For questions or contributions, please refer to the code documentation or open an issue.
