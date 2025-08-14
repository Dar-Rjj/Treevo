import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Price Returns
    df['daily_return'] = df['close'] - df['close'].shift(1)
    
    # Identify Volume Spike Days
    df['volume_20day_ma'] = df['volume'].rolling(window=20).mean()
    
    # Compute Volume Adjusted Momentum
    df['momentum_indicator'] = (df['close'] - df['close'].shift(15)) / df['close'].shift(15)
    df['volume_adjusted_momentum'] = df['momentum_indicator'] * df['volume']
    
    # Combine Price Returns with Volume Spikes
    df['adjusted_daily_return'] = df.apply(lambda row: row['daily_return'] * 2.5 if row['volume'] > 2.0 * row['volume_20day_ma'] else row['daily_return'], axis=1)
    
    # Incorporate True Range for Volatility Adjustment
    df['true_range'] = df.apply(lambda row: max(row['high'] - row['low'], row['high'] - row['close'].shift(1), row['close'].shift(1) - row['low']), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=20).mean()
    df['volatility_adjusted_momentum'] = df['volume_adjusted_momentum'] / df['average_true_range']
    
    # Return the factor values
    return df['volatility_adjusted_momentum'].dropna()
