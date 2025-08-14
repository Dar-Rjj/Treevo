import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generates a High-Low Breakout Momentum factor.
    
    Parameters:
    df (pd.DataFrame): DataFrame with columns ['open', 'high', 'low', 'close', 'amount', 'volume']
                       and index as dates.
                       
    Returns:
    pd.Series: Factor values indexed by date.
    """
    # Calculate High-Low Difference
    df['high_low_diff'] = df['high'] - df['low']
    
    # Calculate Average True Range over a rolling window
    def true_range(row):
        tr1 = row['high'] - row['low']
        tr2 = abs(row['high'] - row['close'].shift(1))
        tr3 = abs(row['low'] - row['close'].shift(1))
        return max(tr1, tr2, tr3)
    
    df['true_range'] = df.apply(true_range, axis=1)
    df['atr'] = df['true_range'].rolling(window=14).mean()
    
    # Adjust for recent volatility using Exponential Moving Average of the ATR
    df['atr_ema'] = df['atr'].ewm(span=14, adjust=False).mean()
    
    # Identify Breakout Days
    df['breakout'] = df['high_low_diff'] > 2 * df['atr_ema']
    
    # Generate Breakout Signal
    df['signal'] = 0
    df.loc[df['breakout'] & (df['close'] > df['open']), 'signal'] = 1  # Positive Breakout
    df.loc[df['breakout'] & (df['close'] < df['open']), 'signal'] = -1  # Negative Breakout
    
    # Return the breakout signal as the factor
    return df['signal'].rename('breakout_momentum_factor')
