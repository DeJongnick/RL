"""Training script for trading agents (DQN, PPO, etc.)."""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.dqn_agent import DQNAgent
from src.agents.ppo_agent import PPOAgent
from src.utils.data_loader import load_and_preprocess_data
from src.utils.environment_setup import create_training_env


def resolve_project_path(path_like):
    """Resolve a path relative to project root if it's not absolute."""
    path = Path(path_like)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_config(config_path='configs/hyperparameters_v1.yaml'):
    """Load configuration YAML file."""
    config_path = resolve_project_path(config_path)
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def create_agent(agent_type, state_dim, env_conf, agent_conf, network_conf, device):
    """
    Create an agent based on the agent type.
    
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
            lr=agent_conf['lr'],
            gamma=agent_conf['gamma'],
            epsilon_start=agent_conf['epsilon_start'],
            epsilon_end=agent_conf['epsilon_end'],
            epsilon_decay=agent_conf['epsilon_decay'],
            buffer_size=agent_conf['buffer_size'],
            batch_size=agent_conf['batch_size'],
            target_update_freq=agent_conf['target_update_freq'],
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


def train_agent(config, device=None, agent_type=None):
    """
    Train a single agent according to the configuration.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        agent_type: Type of agent to train ('dqn' or 'ppo'). If None, uses config['agent_type'] or defaults to 'dqn'
    
    Returns:
        Tuple of (agent, env, run_dir)
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get agent type from parameter or config (default to 'dqn' for backward compatibility)
    if agent_type is None:
        agent_type = config.get('agent_type', 'dqn').lower()
    else:
        agent_type = agent_type.lower()
    
    print(f"Device: {device}")
    print(f"Agent type: {agent_type.upper()}")

    # Create unique run directory with timestamp (or use existing if provided)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_base_dir = resolve_project_path(config.get('logging', {}).get('log_dir', 'logs'))
    runs_dir = logs_base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if run_dir is provided in config (for multi-agent training)
    run_dir = config.get('_run_dir')
    if run_dir is None:
        run_dir = runs_dir / f"run_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = Path(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for this run
    checkpoint_dir = run_dir / "checkpoints" / agent_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir = run_dir / "tensorboard" / agent_type
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    csv_log_path = run_dir / f"training_log_{agent_type}.csv"
    config_save_path = run_dir / "config.yaml"
    summary_path = run_dir / f"training_summary_{agent_type}.json"
    
    print(f"\n{'='*60}")
    print(f"Starting training run: {run_dir.name} - {agent_type.upper()}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*60}\n")
    
    # Save the config used for this run
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to {config_save_path}")

    # Load data
    data_conf = config['data']
    data_folder = resolve_project_path(data_conf['data_folder'])
    _, df_train, _ = load_and_preprocess_data(
        data_folder=str(data_folder),
        download_if_missing=data_conf.get('download_if_missing', True)
    )

    # Create environment
    env_conf = config['environment']
    env = create_training_env(
        df_train=df_train,
        positions=env_conf['positions'],
        trading_fees=env_conf['trading_fees'],
        borrow_interest_rate=env_conf['borrow_interest_rate']
    )

    obs, _ = env.reset()
    state_dim = len(obs)
    
    # Create agent
    agent_conf = config['agent']
    network_conf = config.get('network', {})
    
    agent = create_agent(
        agent_type=agent_type,
        state_dim=state_dim,
        env_conf=env_conf,
        agent_conf=agent_conf,
        network_conf=network_conf,
        device=device
    )

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    training_log = []

    train_conf = config['training']
    num_episodes = train_conf['num_episodes']
    max_steps = train_conf['max_steps_per_episode']
    min_buffer_size = train_conf.get('min_buffer_size', 1000)  # For DQN
    eval_frequency = train_conf['eval_frequency']
    save_frequency = train_conf['save_frequency']
    log_frequency = train_conf['log_frequency']
    
    # PPO-specific: update frequency (update every N episodes or after rollout collection)
    ppo_update_frequency = train_conf.get('ppo_update_frequency', 1)  # Update every episode by default

    best_reward = float('-inf')

    for ep in tqdm(range(num_episodes), desc="Training"):
        obs, info = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        ep_loss = 0.0
        ep_q_value = 0.0
        ep_policy_loss = 0.0
        ep_value_loss = 0.0
        ep_entropy = 0.0
        steps = 0
        
        # Reset agent for new episode (important for PPO to clear buffer)
        agent.reset()

        while not done and not truncated:
            # Agent returns position (float), but env.step expects action index
            position = agent.act(obs, training=True)
            # Convert position to action index
            try:
                action = env_conf['positions'].index(position)
            except ValueError:
                # If exact match not found, find closest position
                action = min(range(len(env_conf['positions'])), 
                           key=lambda i: abs(env_conf['positions'][i] - position))
            
            next_obs, reward, done, truncated, info = env.step(action)
            agent.store_transition(obs, position, reward, next_obs, done)

            # DQN: update during episode if buffer is large enough
            if agent_type == 'dqn':
                if hasattr(agent, 'replay_buffer') and len(agent.replay_buffer) >= min_buffer_size:
                    loss = agent.update()
                    if loss is not None:
                        ep_loss += loss
                        with torch.no_grad():
                            state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                            ep_q_value += agent.q_network(state_t).max().item()
            
            obs = next_obs
            ep_reward += reward
            steps += 1
            if steps >= max_steps:
                break

        # PPO: update at end of episode (or every N episodes)
        if agent_type == 'ppo':
            if (ep + 1) % ppo_update_frequency == 0:
                # Get value for last state if episode didn't terminate
                last_value = 0.0
                last_done = done or truncated
                if not last_done:
                    with torch.no_grad():
                        state_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
                        _, _, _, value = agent.network.get_action_and_value(state_t)
                        last_value = value.item()
                
                metrics = agent.update(last_value=last_value, last_done=last_done)
                if metrics is not None:
                    ep_loss += metrics.get('loss', 0.0)
                    ep_policy_loss += metrics.get('policy_loss', 0.0)
                    ep_value_loss += metrics.get('value_loss', 0.0)
                    ep_entropy += metrics.get('entropy', 0.0)
        
        # DQN: decay epsilon
        if agent_type == 'dqn' and hasattr(agent, 'decay_epsilon'):
            agent.decay_epsilon()
        
        # Calculate averages
        num_updates = max(1, steps // ppo_update_frequency if agent_type == 'ppo' else steps)
        avg_loss = ep_loss / num_updates if ep_loss > 0 else 0.0
        avg_q = ep_q_value / max(steps, 1)
        avg_policy_loss = ep_policy_loss / num_updates if ep_policy_loss > 0 else 0.0
        avg_value_loss = ep_value_loss / num_updates if ep_value_loss > 0 else 0.0
        avg_entropy = ep_entropy / num_updates if ep_entropy > 0 else 0.0

        # Get metrics from environment
        metrics = env.get_metrics() if hasattr(env, 'get_metrics') else {}
        try:
            port_return = float(metrics.get("Portfolio Return", "0%").strip('%')) / 100.0
            market_return = float(metrics.get("Market Return", "0%").strip('%')) / 100.0
        except Exception:
            port_return, market_return = 0.0, 0.0

        if ep % log_frequency == 0:
            writer.add_scalar('Training/Episode_Reward', ep_reward, ep)
            writer.add_scalar('Training/Average_Loss', avg_loss, ep)
            writer.add_scalar('Training/Portfolio_Return', port_return, ep)
            writer.add_scalar('Training/Market_Return', market_return, ep)
            writer.add_scalar('Training/Excess_Return', port_return - market_return, ep)
            writer.add_scalar('Training/Steps', steps, ep)
            
            if agent_type == 'dqn':
                writer.add_scalar('Training/Average_Q_Value', avg_q, ep)
                writer.add_scalar('Training/Epsilon', agent.epsilon, ep)
            elif agent_type == 'ppo':
                writer.add_scalar('Training/Policy_Loss', avg_policy_loss, ep)
                writer.add_scalar('Training/Value_Loss', avg_value_loss, ep)
                writer.add_scalar('Training/Entropy', avg_entropy, ep)

        log_entry = {
            'episode': ep + 1,
            'reward': ep_reward,
            'loss': avg_loss,
            'portfolio_return': port_return,
            'market_return': market_return,
            'excess_return': port_return - market_return,
            'steps': steps
        }
        
        if agent_type == 'dqn':
            log_entry['q_value'] = avg_q
            log_entry['epsilon'] = agent.epsilon
        elif agent_type == 'ppo':
            log_entry['policy_loss'] = avg_policy_loss
            log_entry['value_loss'] = avg_value_loss
            log_entry['entropy'] = avg_entropy
        
        training_log.append(log_entry)

        if (ep + 1) % eval_frequency == 0:
            if agent_type == 'dqn':
                print(f"\nEp {ep + 1}: Reward={ep_reward:.2f} | Portfolio={port_return:.2%} | Market={market_return:.2%} | Eps={agent.epsilon:.3f}")
            elif agent_type == 'ppo':
                print(f"\nEp {ep + 1}: Reward={ep_reward:.2f} | Portfolio={port_return:.2%} | Market={market_return:.2%} | Loss={avg_loss:.4f}")

        if (ep + 1) % save_frequency == 0:
            agent.save(checkpoint_dir / f"checkpoint_episode_{ep + 1}.pt")
        if ep_reward > best_reward:
            best_reward = ep_reward
            agent.save(checkpoint_dir / "best_model.pt")

    # Save final model
    final_model_path = checkpoint_dir / "final_model.pt"
    agent.save(final_model_path)
    print(f"\nFinal model saved to {final_model_path}")
    
    # Save best model (already saved during training, but ensure it's there)
    best_model_path = checkpoint_dir / "best_model.pt"
    if not best_model_path.exists():
        agent.save(best_model_path)
    print(f"Best model saved to {best_model_path}")
    
    # Save training log CSV
    pd.DataFrame(training_log).to_csv(csv_log_path, index=False)
    print(f"Training log saved to {csv_log_path}")
    
    # Calculate and save training summary
    df_log = pd.DataFrame(training_log)
    training_summary = {
        'run_id': run_dir.name,
        'timestamp': timestamp,
        'start_time': timestamp,
        'num_episodes': num_episodes,
        'device': str(device),
        'final_metrics': {
            'final_reward': float(df_log['reward'].iloc[-1]),
            'final_portfolio_return': float(df_log['portfolio_return'].iloc[-1]),
            'final_market_return': float(df_log['market_return'].iloc[-1]),
            'final_excess_return': float(df_log['excess_return'].iloc[-1]),
        },
        'best_metrics': {
            'best_reward': float(df_log['reward'].max()),
            'best_episode': int(df_log.loc[df_log['reward'].idxmax(), 'episode']),
            'best_portfolio_return': float(df_log.loc[df_log['reward'].idxmax(), 'portfolio_return']),
            'best_excess_return': float(df_log.loc[df_log['reward'].idxmax(), 'excess_return']),
        },
        'average_metrics': {
            'avg_reward': float(df_log['reward'].mean()),
            'avg_portfolio_return': float(df_log['portfolio_return'].mean()),
            'avg_market_return': float(df_log['market_return'].mean()),
            'avg_excess_return': float(df_log['excess_return'].mean()),
            'avg_loss': float(df_log['loss'].mean()),
        },
        'files': {
            'config': str(config_save_path.relative_to(PROJECT_ROOT)),
            'final_model': str(final_model_path.relative_to(PROJECT_ROOT)),
            'best_model': str(best_model_path.relative_to(PROJECT_ROOT)),
            'training_log': str(csv_log_path.relative_to(PROJECT_ROOT)),
            'tensorboard_dir': str(tensorboard_dir.relative_to(PROJECT_ROOT)),
        }
    }
    
    with open(summary_path, 'w') as f:
        json.dump(training_summary, f, indent=2)
    print(f"Training summary saved to {summary_path}")
    
    writer.close()
    
    print(f"\n{'='*60}")
    print(f"Training completed successfully!")
    print(f"Run directory: {run_dir}")
    print(f"Final reward: {training_summary['final_metrics']['final_reward']:.2f}")
    print(f"Best reward: {training_summary['best_metrics']['best_reward']:.2f} (episode {training_summary['best_metrics']['best_episode']})")
    print(f"Final portfolio return: {training_summary['final_metrics']['final_portfolio_return']:.2%}")
    print(f"{'='*60}\n")
    
    return agent, env, run_dir


def train_all_agents(config, device=None, agent_types=None):
    """
    Train all specified agents with the same configuration.
    
    Args:
        config: Configuration dictionary
        device: PyTorch device
        agent_types: List of agent types to train (default: ['dqn', 'ppo'])
    
    Returns:
        Dictionary mapping agent types to (agent, env, run_dir) tuples
    """
    if agent_types is None:
        agent_types = ['dqn', 'ppo']
    
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*80}")
    print(f"TRAINING ALL AGENTS: {', '.join([a.upper() for a in agent_types])}")
    print(f"{'='*80}\n")
    
    # Create a single run directory for all agents
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_base_dir = resolve_project_path(config.get('logging', {}).get('log_dir', 'logs'))
    runs_dir = logs_base_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    
    run_dir = runs_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save the config used for this run
    config_save_path = run_dir / "config.yaml"
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    print(f"Configuration saved to {config_save_path}")
    
    # Store run_dir in config for individual agent training
    config['_run_dir'] = str(run_dir)
    
    results = {}
    
    # Train each agent
    for agent_type in agent_types:
        print(f"\n{'='*80}")
        print(f"Training {agent_type.upper()} agent...")
        print(f"{'='*80}\n")
        
        try:
            agent, env, _ = train_agent(config, device=device, agent_type=agent_type)
            results[agent_type] = (agent, env, run_dir)
            print(f"\n✓ {agent_type.upper()} training completed successfully!")
        except Exception as e:
            print(f"\n✗ Error training {agent_type.upper()}: {e}")
            import traceback
            traceback.print_exc()
            results[agent_type] = None
    
    # Remove temporary run_dir from config
    if '_run_dir' in config:
        del config['_run_dir']
    
    print(f"\n{'='*80}")
    print(f"All agents training completed!")
    print(f"Run directory: {run_dir}")
    print(f"{'='*80}\n")
    
    return results, run_dir


def main():
    parser = argparse.ArgumentParser(description='Train trading agent (DQN, PPO, etc.)')
    parser.add_argument('--config', type=str, default='configs/hyperparameters_v1.yaml',
                        help='Path to configuration file (relative to project root)')
    parser.add_argument('--device', type=str, default=None, help='Device to use: cuda/cpu')
    parser.add_argument('--all-agents', action='store_true', 
                        help='Train all agents (DQN and PPO) with the same config')
    parser.add_argument('--agents', type=str, nargs='+', default=None,
                        help='List of agent types to train (e.g., dqn ppo)')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(args.device) if args.device else None
    
    if args.all_agents or args.agents:
        agent_types = args.agents if args.agents else None
        train_all_agents(config, device=device, agent_types=agent_types)
    else:
        train_agent(config, device=device)


if __name__ == '__main__':
    main()

