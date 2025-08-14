import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'] / df['close'].shift(1) - 1
    
    # Compute 20-Day Sum of Upward and Downward Returns
    df['up_return_sum_20'] = df[df['returns'] > 0]['returns'].rolling(window=20).sum()
    df['down_return_sum_20'] = df[df['returns'] < 0]['returns'].abs().rolling(window=20).sum()
    
    # Calculate Relative Strength
    df['relative_strength'] = df['up_return_sum_20'] / df['down_return_sum_20']
    
    # Smooth with Exponential Moving Average on Volume
    df['volume_ema_21'] = df['volume'].ewm(span=21, adjust=False).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['volume_ema_21']
    
    # Adjust Relative Strength with Price Momentum
    df['roc_21'] = (df['close'] - df['close'].shift(21)) / df['close'].shift(21)
    df['price_momentum_adjustment'] = (df['close'] - df['roc_21']) / df['close']
    df['adjusted_relative_strength'] = df['smoothed_relative_strength'] * df['price_momentum_adjustment']
    
    # Calculate Short-Term and Long-Term Momentum
    df['short_term_momentum'] = df['close'] - df['close'].shift(15)
    df['long_term_momentum'] = df['close'] - df['close'].shift(70)
    
    # Calculate Short-Term and Long-Term Volatility
    df['daily_returns'] = df['close'].pct_change()
    df['short_term_volatility'] = df['daily_returns'].rolling(window=15).std()
    df['long_term_volatility'] = df['daily_returns'].rolling(window=70).std()
    
    # Combine Momentum and Volatility
    df['short_term_mv'] = df['short_term_momentum'] + df['short_term_volatility']
    df['long_term_mv'] = df['long_term_momentum'] + df['long_term_volatility']
    
    # Final Alpha Factor
    df['alpha_factor'] = (df['short_term_mv'] - df['long_term_mv']) * df['adjusted_relative_strength']
    
    # Enhance Alpha Factor
    df['high_low_ratio_10'] = df['high'].rolling(window=10).max() / df['low'].rolling(window=10).min()
    df['enhanced_alpha_factor'] = df['alpha_factor'] + df['high_low_ratio_10']
    
    return df['enhanced_alpha_factor'].dropna()
