import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Close Price Returns
    df['daily_return'] = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    
    # Aggregate Momentum
    df['momentum_5d'] = df['daily_return'].rolling(window=5).sum()
    df['momentum_10d'] = df['daily_return'].rolling(window=10).sum()
    
    # Calculate Daily Volume Change
    df['volume_change'] = (df['volume'] - df['volume'].shift(1)) / df['volume'].shift(1)
    
    # Aggregated Volume Spike
    df['volume_spike_5d'] = df['volume_change'].rolling(window=5).sum()
    df['volume_spike_10d'] = df['volume_change'].rolling(window=10).sum()
    
    # Calculate Daily Close Price Absolute Returns
    df['abs_daily_return'] = (df['close'] - df['close'].shift(1)).abs() / df['close'].shift(1)
    
    # Aggregate Volatility
    df['volatility_5d'] = df['abs_daily_return'].rolling(window=5).sum()
    df['volatility_10d'] = df['abs_daily_return'].rolling(window=10).sum()
    
    # Combine Momentum, Volume Spike, and Volatility
    df['combined_5d'] = df['momentum_5d'] * df['volume_spike_5d']
    df['combined_10d'] = df['momentum_10d'] * df['volume_spike_10d']
    
    # Filter Positive Interactions
    df['combined_5d'] = df['combined_5d'].apply(lambda x: x if x > 0 else 0)
    df['combined_10d'] = df['combined_10d'].apply(lambda x: x if x > 0 else 0)
    
    # Adjust for Volatility
    df['adjusted_5d'] = df['combined_5d'] / df['volatility_5d']
    df['adjusted_10d'] = df['combined_10d'] / df['volatility_10d']
    
    # Handle Division by Zero
    df['adjusted_5d'] = df['adjusted_5d'].replace([pd.NP.inf, -pd.NP.inf], 0.0)
    df['adjusted_10d'] = df['adjusted_10d'].replace([pd.NP.inf, -pd.NP.inf], 0.0)
    
    # Weighted Average
    weight_5d = 0.6
    weight_10d = 0.4
    df['alpha_factor'] = weight_5d * df['adjusted_5d'] + weight_10d * df['adjusted_10d']
    
    return df['alpha_factor']
