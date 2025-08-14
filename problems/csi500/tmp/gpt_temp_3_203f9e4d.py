import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Short-Term Price Momentum
    short_momentum = df['close'].rolling(window=10).mean() - df['close']
    
    # Calculate Medium-Term Price Momentum
    medium_momentum = df['close'].rolling(window=30).mean() - df['close']
    
    # Calculate Long-Term Price Momentum
    long_momentum = df['close'].rolling(window=50).mean() - df['close']
    
    # Combine Multi-Period Momenta
    combined_momentum = short_momentum + medium_momentum + long_momentum
    
    # Calculate Volume-Weighted Average Return
    daily_returns = df['close'] / df['open'] - 1
    volume_weighted_returns = (daily_returns * df['volume']).sum() / df['volume'].sum()
    
    # Adjust Combined Momentum by Volume-Weighted Average Return
    adjusted_combined_momentum = combined_momentum * volume_weighted_returns
    
    # Assess Trend Following Potential
    fifty_day_ma = df['close'].rolling(window=50).mean()
    trend_weight = (fifty_day_ma > df['close']).astype(int) + 0.5  # 1 if above, 0.5 if not
    
    # Determine Preliminary Factor Value
    preliminary_factor_value = adjusted_combined_momentum + trend_weight * combined_momentum
    
    # Calculate Short-Term Dynamic Volatility
    short_volatility = (df['high'] - df['low']).rolling(window=10).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    
    # Calculate Medium-Term Dynamic Volatility
    medium_volatility = (df['high'] - df['low']).rolling(window=30).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    
    # Calculate Long-Term Dynamic Volatility
    long_volatility = (df['high'] - df['low']).rolling(window=50).apply(lambda x: (x * df.loc[x.index, 'volume']).sum() / df.loc[x.index, 'volume'].sum(), raw=False)
    
    # Combine Multi-Period Dynamic Volatilities
    combined_volatility = short_volatility + medium_volatility + long_volatility
    
    # Adjust Preliminary Factor Value by Combined Dynamic Volatility
    final_factor_value = preliminary_factor_value / combined_volatility
    
    return final_factor_value
