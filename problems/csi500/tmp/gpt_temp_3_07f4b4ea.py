import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Price Momentum
    short_mom = df['close'] - df['close'].rolling(window=10).mean()
    
    # Calculate Medium-Term Price Momentum
    medium_mom = df['close'] - df['close'].rolling(window=30).mean()
    
    # Calculate Long-Term Price Momentum
    long_mom = df['close'] - df['close'].rolling(window=50).mean()
    
    # Combine Multi-Period Momenta
    combined_mom = short_mom + medium_mom + long_mom
    
    # Calculate Volume-Weighted Average Return
    daily_returns = (df['close'] - df['open']) / df['open']
    volume_weighted_returns = (daily_returns * df['volume']).sum() / df['volume'].sum()
    
    # Adjust Combined Momentum by Volume-Weighted Average Return
    adjusted_combined_mom = combined_mom * volume_weighted_returns
    
    # Assess Trend Following Potential
    trend_following_50day_ma = df['close'].rolling(window=50).mean()
    trend_following_weight = 1 if trend_following_50day_ma > df['close'] else 0.5
    trend_following_component = adjusted_combined_mom * trend_following_weight
    
    # Determine Preliminary Factor Value
    preliminary_factor_value = adjusted_combined_mom + trend_following_component
    
    # Calculate Short-Term Volatility
    short_vol = (df['high'] - df['low']).rolling(window=10).mean()
    
    # Calculate Medium-Term Volatility
    medium_vol = (df['high'] - df['low']).rolling(window=30).mean()
    
    # Calculate Long-Term Volatility
    long_vol = (df['high'] - df['low']).rolling(window=50).mean()
    
    # Combine Multi-Period Volatilities
    combined_vol = short_vol + medium_vol + long_vol
    
    # Adjust Combined Momentum by Combined Volatility
    final_adjusted_momentum = adjusted_combined_mom / combined_vol
    
    # Re-evaluate Final Factor Value
    final_factor_value = final_adjusted_momentum + trend_following_component
    
    return final_factor_value
