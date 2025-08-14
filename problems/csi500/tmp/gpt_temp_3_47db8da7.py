import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Volume-Weighted Close-to-Open Return
    df['close_to_open_return'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    df['volume_weighted_return'] = df['close_to_open_return'] * df['volume']
    
    # Calculate Dynamic Lookback Period using ATR
    df['high_low'] = df['high'] - df['low']
    df['high_close_prev'] = np.abs(df['high'] - df['close'].shift(1))
    df['low_close_prev'] = np.abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=20).mean()
    scaling_factor = 10  # this can be adjusted based on the specific market and data characteristics
    df['lookback_period'] = (df['average_true_range'] * scaling_factor).astype(int)

    # Compute Momentum
    def dynamic_moving_average(x, window):
        return x.rolling(window=int(window)).mean()

    df['moving_average'] = df.groupby('date')['close'].apply(lambda x: dynamic_moving_average(x, df.loc[x.index, 'lookback_period']))
    df['momentum'] = df['close'] - df['moving_average']

    # Generate Alpha Factor
    df['alpha_factor'] = df['volume_weighted_return'] + df['momentum']

    return df['alpha_factor'].copy()
