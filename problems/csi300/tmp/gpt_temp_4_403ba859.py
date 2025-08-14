import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Raw Returns
    df['returns'] = df['close'].pct_change()
    
    # Compute 20-Day Sum of Upward and Downward Returns
    up_returns = df[df['returns'] > 0]['returns']
    down_returns = df[df['returns'] < 0]['returns'].abs()
    
    df['up_sum_20'] = up_returns.rolling(window=20).sum()
    df['down_sum_20'] = down_returns.rolling(window=20).sum()
    
    # Calculate Relative Strength
    df['relative_strength'] = df['up_sum_20'] / df['down_sum_20']
    
    # Smooth with Exponential Moving Average on Volume
    df['ema_volume'] = df['volume'].ewm(span=20, adjust=False).mean()
    df['smoothed_relative_strength'] = df['relative_strength'] * df['ema_volume']
    
    # Calculate Short-Term Momentum
    df['short_momentum'] = (df['close'] - df['close'].shift(15)).abs()
    
    # Calculate Long-Term Momentum
    df['long_momentum'] = (df['close'] - df['close'].shift(70)).abs()
    
    # Calculate Short-Term Volatility
    df['short_volatility'] = df['close'].pct_change().rolling(window=15).std().abs()
    
    # Calculate Long-Term Volatility
    df['long_volatility'] = df['close'].pct_change().rolling(window=70).std().abs()
    
    # Combine Momentum and Volatility
    df['short_mom_vol'] = df['short_momentum'] * df['short_volatility']
    df['long_mom_vol'] = df['long_momentum'] * df['long_volatility']
    
    # Final Alpha Factor
    df['alpha_factor'] = df['short_mom_vol'] - df['long_mom_vol']
    
    # Enhance Alpha Factor
    df['enhanced_alpha_factor'] = df['alpha_factor'] * df['smoothed_relative_strength']
    
    return df['enhanced_alpha_factor'].dropna()
