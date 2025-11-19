"""Utilities for data loading and preprocessing."""

import pandas as pd
from pathlib import Path
from datetime import datetime
from gym_trading_env.downloader import download


def download_data(
    data_folder,
    exchange_names=["binance"],
    symbols=["BTC/USDT"],
    timeframe="1h",
    since=None,
):
    """
    Downloads historical market data if missing.
    Returns the path to the data file.
    
    Args:
        data_folder: Directory to save data
        exchange_names: List of exchange names
        symbols: List of trading pairs
        timeframe: Timeframe for data (e.g., "1h")
        since: Start date for data download
    
    Returns:
        Path to the downloaded data file
    """
    data_folder = Path(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)
    since = since or datetime(year=2020, month=10, day=1)
    download(
        exchange_names=exchange_names,
        symbols=symbols,
        timeframe=timeframe,
        dir=data_folder,
        since=since,
    )
    filename = f"{exchange_names[0]}-{symbols[0].replace('/', '')}-{timeframe}.pkl"
    return data_folder / filename


def preprocess_data(df):
    """
    Minimal preprocessing: create time-series features from an OHLCV DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
    
    Returns:
        Preprocessed DataFrame with features
    """
    df = df.copy()
    df["feature_close"] = df["close"].pct_change()
    df["feature_open"] = df["open"] / df["close"]
    df["feature_high"] = df["high"] / df["close"]
    df["feature_low"] = df["low"] / df["close"]
    df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
    df.dropna(inplace=True)  # Remove initial incomplete values
    return df


def split_data(
    df,
    train_start='2024-10-01',
    train_end='2025-09-30',
    eval_start='2025-10-01',
    eval_end='2025-11-01',
):
    """
    Split the data between train and eval based on ISO date ranges.
    
    Args:
        df: Full DataFrame
        train_start: Start date for training data
        train_end: End date for training data
        eval_start: Start date for evaluation data
        eval_end: End date for evaluation data
    
    Returns:
        Tuple of (df_train, df_eval)
    """
    df_train = df.loc[train_start:train_end]
    df_eval = df.loc[eval_start:eval_end]
    return df_train, df_eval


def load_and_preprocess_data(data_folder, download_if_missing=True):
    """
    Load and prepare the data. Download if needed.
    The relative data folder path is resolved from the project root.
    
    Args:
        data_folder: Path to data folder (relative or absolute)
        download_if_missing: Whether to download data if file is missing
    
    Returns:
        Tuple of (df, df_train, df_eval)
    """
    data_folder = Path(data_folder)
    if not data_folder.is_absolute():
        # Ensure consistent path relative to project root
        project_root = Path(__file__).resolve().parent.parent.parent
        data_folder = project_root / data_folder
    data_file = data_folder / "binance-BTCUSDT-1h.pkl"

    if not data_file.exists():
        if download_if_missing:
            print(f"Downloading data to {data_file} ...")
            data_file = download_data(data_folder)
        else:
            raise FileNotFoundError(f"Data file not found: {data_file}")

    print(f"Loading: {data_file} ...")
    df = pd.read_pickle(data_file)

    print("Preprocessing ...")
    df = preprocess_data(df)

    print("Splitting train/eval ...")
    df_train, df_eval = split_data(df)

    print(f"Train data: {len(df_train)} ({df_train.index[0]} -> {df_train.index[-1]})")
    print(f"Eval data : {len(df_eval)} ({df_eval.index[0]} -> {df_eval.index[-1]})")
    return df, df_train, df_eval

