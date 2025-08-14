import pandas as pd
import pandas as pd

def heuristics(df):
    # Calculate Daily Price Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Volume Relative Strength
    df['volume_20d_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_relative_strength'] = df['volume'] / df['volume_20d_avg']
    
    # Calculate Weighted Average Price
    df['weighted_avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    
    # Calculate Momentum
    df['momentum'] = df['weighted_avg_price'].pct_change(periods=10)
    df['smoothed_momentum'] = df['momentum'].rolling(window=5).mean()
    
    # Identify Volume Spikes
    volume_spike_threshold = 2.0  # Example threshold
    df['volume_spike'] = df['volume'] > (volume_spike_threshold * df['volume_20d_avg'])
    
    # Adjust Momentum by Volume Spike
    scaling_factor_volume_spike = 1.5  # Example scaling factor
    df['adjusted_momentum'] = df.apply(
        lambda row: row['smoothed_momentum'] * scaling_factor_volume_spike if row['volume_spike'] else row['smoothed_momentum'],
        axis=1
    )
    
    # Integrate Price Volatility
    df['true_range'] = df[['high', 'low']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['avg_true_range'] = df['true_range'].rolling(window=10).mean()
    
    # Adjust Momentum by Price Volatility Spike
    volatility_spike_threshold = 2.0  # Example threshold
    df['volatility_spike'] = df['true_range'] > (volatility_spike_threshold * df['avg_true_range'])
    
    scaling_factor_volatility_spike = 1.2  # Example scaling factor
    df['final_adjusted_momentum'] = df.apply(
        lambda row: row['adjusted_momentum'] * scaling_factor_volatility_spike if row['volatility_spike'] else row['adjusted_momentum'],
        axis=1
    )
    
    # Calculate Final Factor
    df['final_factor'] = df['final_adjusted_momentum'] * df['price_range']
    
    return df['final_factor']
