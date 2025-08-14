import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_return'] = df['close'].pct_change()
    
    # Long-Term Volume-Weighted Average Return (Momentum Component)
    def long_term_vol_weighted_return(df, window=100):
        weighted_returns = df['daily_return'].rolling(window=window).apply(
            lambda x: (x * df['volume'].shift(window - len(x) + 1)).sum(), raw=False
        )
        total_volume = df['volume'].rolling(window=window).sum()
        return weighted_returns / total_volume
    
    # Short-Term Volume-Weighted Average Return (Reversal Component)
    def short_term_vol_weighted_return(df, window=5):
        weighted_returns = df['daily_return'].rolling(window=window).apply(
            lambda x: (x * df['volume'].shift(window - len(x) + 1)).sum(), raw=False
        )
        total_volume = df['volume'].rolling(window=window).sum()
        return weighted_returns / total_volume
    
    # Calculate the momentum and reversal components
    df['long_term_vol_weighted_return'] = long_term_vol_weighted_return(df)
    df['short_term_vol_weighted_return'] = short_term_vol_weighted_return(df)
    
    # Combine Momentum and Reversal Components
    df['alpha_factor'] = df['long_term_vol_weighted_return'] - df['short_term_vol_weighted_return']
    
    return df['alpha_factor']
