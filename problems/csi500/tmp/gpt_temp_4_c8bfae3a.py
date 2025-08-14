import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, alpha=0.1, V_threshold=1.5, P=0.01, N=14):
    # Calculate daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Initialize EMA with the first daily return
    df['ema'] = 0
    df.loc[df.index[0], 'ema'] = df.loc[df.index[0], 'daily_return']
    
    # Calculate EMA for daily returns
    for i in range(1, len(df)):
        df.loc[df.index[i], 'ema'] = alpha * df.loc[df.index[i], 'daily_return'] + (1 - alpha) * df.loc[df.index[i-1], 'ema']
    
    # Apply enhanced volume filter
    df['volume_ratio'] = df['volume'] / df['volume'].shift(1)
    df['ema_filtered'] = df['ema'].where(df['volume_ratio'] > V_threshold, 0)
    
    # Calculate True Range
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['close'].shift(1)).abs()
    df['tr3'] = (df['low'] - df['close'].shift(1)).abs()
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Calculate ATR
    df['atr'] = df['true_range'].rolling(window=N).mean()
    
    # Apply enhanced price volatility filter
    df['factor'] = df['ema_filtered'].where(df['atr'] > P, 0)
    
    return df['factor']

# Example usage:
# df = pd.read_csv('your_data.csv', index_col='date', parse_dates=True)
# factor_values = heuristics_v2(df)
# print(factor_values)
