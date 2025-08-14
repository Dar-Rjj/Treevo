import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Returns
    df['raw_returns'] = df['close'].pct_change()
    
    # Compute 20-Day Sum of Upward and Downward Returns
    df['up_returns'] = df['raw_returns'].apply(lambda x: x if x > 0 else 0)
    df['down_returns'] = df['raw_returns'].apply(lambda x: abs(x) if x < 0 else 0)
    df['sum_up_returns_20'] = df['up_returns'].rolling(window=20).sum()
    df['sum_down_returns_20'] = df['down_returns'].rolling(window=20).sum()
    
    # Calculate Relative Strength
    df['relative_strength'] = df['sum_up_returns_20'] / df['sum_down_returns_20']
    
    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=20, adjust=False, alpha=0.3).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']
    
    # Calculate High-Low Spread
    df['high_low_spread'] = df['high'] - df['low']
    
    # Calculate High-Low Mean Reversion
    df['avg_high_low_spread_20'] = df['high_low_spread'].rolling(window=20).mean()
    df['high_low_mean_reversion'] = df['high_low_spread'] - df['avg_high_low_spread_20']
    
    # Calculate Close Price Momentum
    df['close_momentum_20'] = df['close'].pct_change(periods=20)
    
    # Combine Factors
    df['combined_factors'] = (df['high_low_mean_reversion'] + df['close_momentum_20']) / df['high_low_spread']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['smoothed_relative_strength'] * df['combined_factors']
    
    return df['alpha_factor']
