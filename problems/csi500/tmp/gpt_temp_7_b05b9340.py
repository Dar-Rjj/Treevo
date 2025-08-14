import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate High-Low Spread
    high_low_spread = df['high'] - df['low']
    
    # Calculate Volume-Average Price
    volume_average_price = (df['close'] * df['volume']).rolling(window=1).sum() / df['volume'].rolling(window=1).sum()
    
    # Compute Ratio: Volume-Averaged High-Low Spread
    volume_averaged_high_low_ratio = high_low_spread / volume_average_price
    
    # Calculate Daily Returns
    daily_returns = df['close'].pct_change()
    
    # Long-Term Volume-Weighted Average Return (Momentum Component)
    long_term_weighted_returns = (daily_returns.rolling(window=100).apply(lambda x: (x * df['volume']).sum(), raw=True) / df['volume'].rolling(window=100).sum()).fillna(0)
    
    # Short-Term Volume-Weighted Average Return (Reversal Component)
    short_term_weighted_returns = (daily_returns.rolling(window=5).apply(lambda x: (x * df['volume']).sum(), raw=True) / df['volume'].rolling(window=5).sum()).fillna(0)
    
    # Combine Factors
    momentum_reversal_factor = long_term_weighted_returns - short_term_weighted_returns
    final_alpha_factor = momentum_reversal_factor * volume_averaged_high_low_ratio
    
    return final_alpha_factor
