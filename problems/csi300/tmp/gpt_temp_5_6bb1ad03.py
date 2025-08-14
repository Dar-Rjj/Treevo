import pandas as pd
import pandas as pd

def heuristics_v2(df, lookback_period=14):
    # Calculate Intraday Range
    df['intraday_range'] = df['high'] - df['low']
    
    # Calculate Close-to-Open Return
    df['close_to_open_return'] = df['close'].pct_change()
    
    # Calculate True Range
    df['true_range'] = df.apply(
        lambda x: max(x['high'] - x['low'], 
                      abs(x['high'] - x['close'].shift(1)), 
                      abs(x['low'] - x['close'].shift(1))), 
        axis=1
    )
    
    # Calculate Average True Range
    df['average_true_range'] = df['true_range'].rolling(window=lookback_period).mean()
    
    # Adjust Close-to-Open Return by Intraday Range
    df['adjusted_close_to_open_return'] = df['close_to_open_return'] / df['intraday_range']
    
    # Adjust for Intraday Volatility (Average True Range)
    df['intraday_volatility_adjusted_momentum'] = df['adjusted_close_to_open_return'] / df['average_true_range']
    
    return df['intraday_volatility_adjusted_momentum'].dropna()
