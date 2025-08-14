import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Log Return
    df['log_return'] = np.log(df['close']).diff()

    # Compute Volume-Weighted Log Returns
    df['volume_weighted_log_return'] = df['log_return'] * df['volume']

    # Take Long-Term Moving Sum of Volume-Weighted Log Returns (e.g., 20 days)
    long_term_window = 20
    df['long_term_momentum'] = df['volume_weighted_log_return'].rolling(window=long_term_window).sum()

    # Calculate Short-Term Volatility (Average True Range over 5 days)
    short_term_window = 5
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x[0] - x[1], abs(x[0] - df['close'].shift(1)), abs(x[1] - df['close'].shift(1))), axis=1)
    df['short_term_volatility'] = df['true_range'].rolling(window=short_term_window).mean()

    # Combine Long-Term Momentum and Short-Term Volatility
    df['alpha_factor'] = df['long_term_momentum'] / df['short_term_volatility']

    return df['alpha_factor']
