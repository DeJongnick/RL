"""Utility script to list and compare training runs."""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def list_runs(logs_dir=None):
    """List all training runs with their summaries."""
    if logs_dir is None:
        logs_dir = PROJECT_ROOT / "logs" / "runs"
    else:
        logs_dir = Path(logs_dir)
    
    if not logs_dir.exists():
        print(f"No runs directory found at {logs_dir}")
        return []
    
    runs = []
    for run_dir in sorted(logs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        
        summary_path = run_dir / "training_summary.json"
        if summary_path.exists():
            try:
                with open(summary_path, 'r') as f:
                    summary = json.load(f)
                runs.append({
                    'run_dir': run_dir,
                    'summary': summary
                })
            except Exception as e:
                print(f"Error reading {summary_path}: {e}")
    
    return runs


def print_runs_table(runs):
    """Print a formatted table of all runs."""
    if not runs:
        print("No training runs found.")
        return
    
    print(f"\n{'='*100}")
    print(f"{'Run ID':<20} {'Timestamp':<20} {'Best Reward':<15} {'Final Return':<15} {'Best Return':<15} {'Episodes':<10}")
    print(f"{'-'*100}")
    
    for run_info in runs:
        summary = run_info['summary']
        run_id = summary['run_id']
        timestamp = summary['timestamp']
        best_reward = summary['best_metrics']['best_reward']
        final_return = summary['final_metrics']['final_portfolio_return']
        best_return = summary['best_metrics']['best_portfolio_return']
        episodes = summary['num_episodes']
        
        print(f"{run_id:<20} {timestamp:<20} {best_reward:<15.2f} {final_return:<15.2%} {best_return:<15.2%} {episodes:<10}")
    
    print(f"{'='*100}\n")


def print_run_details(run_info):
    """Print detailed information about a specific run."""
    summary = run_info['summary']
    run_dir = run_info['run_dir']
    
    print(f"\n{'='*80}")
    print(f"Run Details: {summary['run_id']}")
    print(f"{'='*80}")
    print(f"Timestamp: {summary['timestamp']}")
    print(f"Device: {summary['device']}")
    print(f"Episodes: {summary['num_episodes']}")
    print(f"\nFinal Metrics:")
    print(f"  Reward: {summary['final_metrics']['final_reward']:.2f}")
    print(f"  Portfolio Return: {summary['final_metrics']['final_portfolio_return']:.2%}")
    print(f"  Market Return: {summary['final_metrics']['final_market_return']:.2%}")
    print(f"  Excess Return: {summary['final_metrics']['final_excess_return']:.2%}")
    print(f"  Epsilon: {summary['final_metrics']['final_epsilon']:.4f}")
    print(f"\nBest Metrics:")
    print(f"  Reward: {summary['best_metrics']['best_reward']:.2f} (Episode {summary['best_metrics']['best_episode']})")
    print(f"  Portfolio Return: {summary['best_metrics']['best_portfolio_return']:.2%}")
    print(f"  Excess Return: {summary['best_metrics']['best_excess_return']:.2%}")
    print(f"\nAverage Metrics:")
    print(f"  Reward: {summary['average_metrics']['avg_reward']:.2f}")
    print(f"  Portfolio Return: {summary['average_metrics']['avg_portfolio_return']:.2%}")
    print(f"  Market Return: {summary['average_metrics']['avg_market_return']:.2%}")
    print(f"  Excess Return: {summary['average_metrics']['avg_excess_return']:.2%}")
    print(f"  Loss: {summary['average_metrics']['avg_loss']:.4f}")
    print(f"\nFiles:")
    for key, path in summary['files'].items():
        print(f"  {key}: {path}")
    print(f"\nRun Directory: {run_dir}")
    print(f"{'='*80}\n")


def compare_runs(runs, metric='best_reward', top_n=5):
    """Compare runs by a specific metric."""
    if not runs:
        print("No runs to compare.")
        return
    
    # Sort runs by metric
    metric_map = {
        'best_reward': lambda s: s['best_metrics']['best_reward'],
        'final_reward': lambda s: s['final_metrics']['final_reward'],
        'best_return': lambda s: s['best_metrics']['best_portfolio_return'],
        'final_return': lambda s: s['final_metrics']['final_portfolio_return'],
        'excess_return': lambda s: s['best_metrics']['best_excess_return'],
        'avg_return': lambda s: s['average_metrics']['avg_portfolio_return'],
    }
    
    if metric not in metric_map:
        print(f"Unknown metric: {metric}")
        print(f"Available metrics: {', '.join(metric_map.keys())}")
        return
    
    sorted_runs = sorted(runs, key=lambda r: metric_map[metric](r['summary']), reverse=True)
    
    print(f"\n{'='*100}")
    print(f"Top {min(top_n, len(sorted_runs))} runs by {metric}:")
    print(f"{'='*100}")
    print(f"{'Rank':<6} {'Run ID':<20} {'Timestamp':<20} {'Value':<15}")
    print(f"{'-'*100}")
    
    for i, run_info in enumerate(sorted_runs[:top_n], 1):
        summary = run_info['summary']
        value = metric_map[metric](summary)
        print(f"{i:<6} {summary['run_id']:<20} {summary['timestamp']:<20} {value:<15.4f}")
    
    print(f"{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(description='List and compare training runs')
    parser.add_argument('--logs-dir', type=str, default=None,
                        help='Path to logs directory (default: logs/runs)')
    parser.add_argument('--details', type=str, default=None,
                        help='Show details for a specific run ID')
    parser.add_argument('--compare', type=str, default=None,
                        help='Compare runs by metric (best_reward, final_reward, best_return, etc.)')
    parser.add_argument('--top-n', type=int, default=5,
                        help='Number of top runs to show when comparing (default: 5)')
    args = parser.parse_args()
    
    runs = list_runs(args.logs_dir)
    
    if args.details:
        # Find the specific run
        found = False
        for run_info in runs:
            if run_info['summary']['run_id'] == args.details:
                print_run_details(run_info)
                found = True
                break
        if not found:
            print(f"Run '{args.details}' not found.")
    elif args.compare:
        compare_runs(runs, metric=args.compare, top_n=args.top_n)
    else:
        print_runs_table(runs)


if __name__ == '__main__':
    main()

