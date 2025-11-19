"""Quick check of project installation and structure."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("=== Environment Check ===")
print(f"Python: {sys.version.split()[0]}")

# --- Check for required packages ---
required = [
    'numpy', 'pandas', 'gymnasium', 'gym_trading_env',
    'torch', 'tensorboard', 'matplotlib', 'seaborn', 'yaml', 'tqdm'
]
missing = []
for pkg in required:
    try:
        __import__(pkg if pkg != 'yaml' else 'yaml')
        print(f"  ✓ {pkg}")
    except ImportError:
        print(f"  ✗ {pkg} (missing)")
        missing.append(pkg)
if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("Install them with: pip install -r requirements.txt")
else:
    print("All required packages are present.")

# --- Check key project directories ---
print("\nChecking key directories:")
for d in ['src/agents', 'src/models', 'src/utils', 'configs', 'logs', 'eval', 'data']:
    target = PROJECT_ROOT / d
    print(f"  {'✓' if target.exists() else '✗'} {d}/")

print("\nChecking essential files:")
key_files = [
    'scripts/train.py', 'scripts/evaluate.py', 'requirements.txt', 'README.md',
    'configs/hyperparameters_v1.yaml',
    'src/agents/dqn_agent.py', 'src/models/dqn_network.py',
    'src/utils/data_loader.py', 'src/utils/evaluation.py'
]
for f in key_files:
    target = PROJECT_ROOT / f
    print(f"  {'✓' if target.exists() else '✗'} {f}")

# --- Test imports of important modules ---
print("\nTesting main imports:")
imports = [
    ("src.agents.dqn_agent", "DQNAgent"),
    ("src.models.dqn_network", "DuelingDQN"),
    ("src.utils.data_loader", "load_and_preprocess_data"),
    ("src.utils.evaluation", "evaluate_agent"),
]
for mod, obj in imports:
    try:
        module = __import__(mod, fromlist=[obj])
        getattr(module, obj)
        print(f"  ✓ {mod}")
    except Exception as e:
        print(f"  ✗ {mod} : {e}")

print("\nCheck complete.")
