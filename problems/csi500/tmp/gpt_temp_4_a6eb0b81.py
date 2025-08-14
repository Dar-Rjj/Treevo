import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Define lookback periods
    short_term_lookback = 10
    medium_term_lookback = 30
    n = 5
    m = 5
    percentage_change_lookback = 5
    
    # Calculate Daily High-Low Price Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Dynamically Define the Lookback Period based on Recent Volatility
    df['volatility'] = df['high_low_diff'].rolling(window=short_term_lookback).std()
    lookback_period = (df['volatility'] / df['volatility'].mean() * 20).astype(int).clip(lower=5, upper=60)
    
    # Calculate Dynamic Simple Moving Average (DSMA) of Close Prices
    def dsma(values, window):
        return values.rolling(window=window).mean()
    
    df['dsma'] = [dsma(df['close'], period)[i] for i, period in enumerate(lookback_period)]
    
    # Compute Volume-Adjusted High-Low Volatility
    df['volume_weighted_high_low'] = df['high_low_diff'] * df['volume']
    df['volume_adjusted_volatility'] = df['volume_weighted_high_low'].rolling(window=medium_term_lookback).mean()
    
    # Compute Adaptive Price Momentum
    df['price_momentum'] = (df['close'] - df['dsma']) / df['close'].rolling(window=n).sum()
    
    # Incorporate Additional Price Change Metrics
    df['percentage_change'] = df['close'].pct_change(periods=percentage_change_lookback)
    df['high_low_range'] = (df['high'] - df['low']) / df['close'].rolling(window=m).mean()
    
    # Final Alpha Factor
    weights = {
        'price_momentum': 0.4,
        'volume_adjusted_volatility': 0.3,
        'percentage_change': 0.2,
        'high_low_range': 0.1
    }
    
    df['alpha_factor'] = (
        df['price_momentum'] * weights['price_momentum'] +
        df['volume_adjusted_volatility'] * weights['volume_adjusted_volatility'] +
        df['percentage_change'] * weights['percentage_change'] +
        df['high_low_range'] * weights['high_low_range']
    )
    
    return df['alpha_factor'].dropna()
