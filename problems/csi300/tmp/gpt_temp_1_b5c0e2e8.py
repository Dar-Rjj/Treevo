import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Define Exponentially Decreasing Weights
    def exponential_weights(length, halflife):
        return np.exp(-np.log(2) / halflife * np.arange(length))

    # Compute Cumulative Weighted Return
    lookback_period = 20
    half_life = 5
    weights = exponential_weights(lookback_period, half_life)
    df['weighted_return'] = df['daily_return'].rolling(window=lookback_period).apply(lambda x: (x * weights[:len(x)]).sum(), raw=True)

    # Adjust by Volatility
    df['true_range'] = df[['high', 'low', 'close']].apply(lambda x: max(x['high'] - x['low'], abs(x['high'] - x['close'].shift()), abs(x['low'] - x['close'].shift())), axis=1)
    df['average_true_range'] = df['true_range'].rolling(window=lookback_period).mean()
    df['adjusted_weighted_return'] = df['weighted_return'] / df['average_true_range']

    # Incorporate Volume Impact
    df['volume_change'] = df['volume'].pct_change()
    df['weighted_volume_change'] = df['volume_change'].rolling(window=lookback_period).apply(lambda x: (x * weights[:len(x)]).sum(), raw=True)

    # Incorporate Amount Impact
    df['amount_change'] = df['amount'].pct_change()
    df['weighted_amount_change'] = df['amount_change'].rolling(window=lookback_period).apply(lambda x: (x * weights[:len(x)]).sum(), raw=True)

    # Combine Indicators
    df['alpha_factor'] = df['adjusted_weighted_return'] + df['weighted_volume_change'] + df['weighted_amount_change']

    return df['alpha_factor']
