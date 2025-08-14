import pandas as pd
import pandas as pd

def heuristics_v2(df, n=5, spike_factor=2):
    # Calculate Intraday Return Components
    high_to_open_return = (df['high'] - df['open']) / df['open']
    close_to_low_return = (df['close'] - df['low']) / df['low']
    open_to_low_return = (df['open'] - df['low']) / df['low']
    
    # Aggregate Intraday Return Components
    aggregate_intraday_return = high_to_open_return + close_to_low_return + open_to_low_return
    weighted_aggregate_intraday_return = aggregate_intraday_return * df['volume']
    
    # Cumulative Effect
    cimi = weighted_aggregate_intraday_return.rolling(window=2).sum().shift(1)
    cimi = cimi.fillna(0) + weighted_aggregate_intraday_return
    
    # Calculate Daily Return
    daily_return = df['close'].pct_change()
    
    # Identify Volume Spike
    volume_change = df['volume'].pct_change()
    is_spike = volume_change > spike_factor
    
    # Adjust Momentum by Volume Spike
    n_day_momentum = daily_return.rolling(window=n).sum()
    adjusted_momentum = n_day_momentum * (spike_factor if is_spike else 1)
    
    return cimi * adjusted_momentum
