import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Intraday Momentum
    df['high_low_diff'] = df['high'] - df['low']
    df['intraday_momentum'] = df['high_low_diff'] / df['low']

    # Calculate True Range
    df['true_range'] = df.apply(
        lambda row: max(row['high'] - row['low'], 
                        abs(row['high'] - row['close'].shift(1)), 
                        abs(row['low'] - row['close'].shift(1))), 
        axis=1
    )
    
    # Calculate Average True Range (ATR) over 14 days
    df['atr_14'] = df['true_range'].rolling(window=14).mean()

    # Calculate Volatility Adjusted Return
    df['daily_return'] = (df['close'] - df['open']) / df['open']
    df['volatility_adjusted_return'] = df['daily_return'] / df['atr_14']

    # Combine Intraday Momentum and Volatility Adjusted Return
    df['alpha_factor'] = df['intraday_momentum'] * df['volatility_adjusted_return']

    return df['alpha_factor']
