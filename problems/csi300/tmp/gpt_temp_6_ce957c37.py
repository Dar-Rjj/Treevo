import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'].pct_change()
    
    # Compute 20-Day Sum of Upward and Downward Returns
    df['up_returns'] = df['returns'].apply(lambda x: x if x > 0 else 0)
    df['down_returns'] = df['returns'].apply(lambda x: -x if x < 0 else 0)
    df['up_sum_20'] = df['up_returns'].rolling(window=20).sum()
    df['down_sum_20'] = df['down_returns'].rolling(window=20).sum()
    
    # Calculate Relative Strength
    df['relative_strength'] = df['up_sum_20'] / df['down_sum_20']
    
    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=20, adjust=False, alpha=0.3).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']
    
    # Calculate Short-Term and Long-Term Momentum
    df['short_momentum'] = (df['close'] - df['close'].shift(15)).abs()
    df['long_momentum'] = (df['close'] - df['close'].shift(70)).abs()
    
    # Calculate Short-Term and Long-Term Volatility
    df['daily_returns'] = df['close'].pct_change()
    df['short_volatility'] = df['daily_returns'].rolling(window=15).std().abs()
    df['long_volatility'] = df['daily_returns'].rolling(window=70).std().abs()
    
    # Combine Momentum and Volatility
    df['momentum_volatility_sum'] = df['short_momentum'] + df['short_volatility']
    df['momentum_volatility_product'] = df['long_momentum'] * df['long_volatility']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['momentum_volatility_sum'] - df['momentum_volatility_product']
    
    # Enhance Alpha Factor
    df['alpha_factor_enhanced'] = df['alpha_factor'] * df['smoothed_relative_strength']
    df['high_low_ratio'] = df['high'].rolling(window=10).max() / df['low'].rolling(window=10).min()
    df['final_alpha_factor'] = df['alpha_factor_enhanced'] + df['high_low_ratio']
    
    return df['final_alpha_factor']
