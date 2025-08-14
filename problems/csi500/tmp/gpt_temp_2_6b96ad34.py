import pandas as pd
import pandas as pd

def heuristics_v2(df, n=20, spike_factor=2.5):
    # Calculate Intraday Return Components
    df['high_to_open_return'] = (df['high'] - df['open']) / df['open']
    df['close_to_low_return'] = (df['close'] - df['low']) / df['low']
    df['open_to_close_return'] = (df['close'] - df['open']) / df['open']

    # Aggregate Intraday Return Components
    df['aggregate_intraday_return'] = df['high_to_open_return'] + df['close_to_low_return'] + df['open_to_close_return']
    df['weighted_aggregate_intraday_return'] = df['aggregate_intraday_return'] * df['volume']

    # Cumulative Effect
    df['cimi'] = 0.0
    for i in range(1, len(df)):
        df.loc[df.index[i], 'cimi'] = df.loc[df.index[i-1], 'cimi'] + df.loc[df.index[i], 'weighted_aggregate_intraday_return']

    # Calculate Daily Return
    df['daily_return'] = df['close'].pct_change()

    # Identify Volume Spike
    df['volume_change'] = df['volume'] - df['volume'].shift(1)
    df['volume_spike'] = (df['volume_change'] > spike_factor * df['volume'].shift(1)).astype(int)

    # Calculate N-day Momentum
    df['n_day_momentum'] = df['daily_return'].rolling(window=n).sum()

    # Adjust Momentum by Volume Spike
    df['adjusted_momentum'] = df['n_day_momentum'] * (1 + (spike_factor - 1) * df['volume_spike'])

    return df['adjusted_momentum']
