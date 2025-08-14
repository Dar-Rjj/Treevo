import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Short-Term Volume Adjusted Return
    short_term_window = 5
    df['short_term_return'] = df['close'].pct_change(periods=short_term_window)
    df['short_term_volume_adjusted_return'] = df['short_term_return'] * df['volume']

    # Calculate Long-Term Volume Adjusted Return
    long_term_window = 20
    df['long_term_return'] = df['close'].pct_change(periods=long_term_window)
    df['long_term_volume_adjusted_return'] = df['long_term_return'] * df['volume']

    # Calculate Price Reversal Indicator
    def identify_local_extrema(series, window, kind='high'):
        if kind == 'high':
            return (series.rolling(window=window, center=True).max() == series) & (series.shift(1) < series) & (series.shift(-1) < series)
        elif kind == 'low':
            return (series.rolling(window=window, center=True).min() == series) & (series.shift(1) > series) & (series.shift(-1) > series)
    
    local_highs = identify_local_extrema(df['high'], window=10, kind='high')
    local_lows = identify_local_extrema(df['low'], window=10, kind='low')
    
    df['local_highs'] = local_highs
    df['local_lows'] = local_lows

    def time_since_last_extremum(df, extremum_column):
        last_extremum = df[extremum_column].idxmax()
        return (df.index - last_extremum).days
    
    df['time_since_last_high'] = df.groupby(df['local_highs'].cumsum().shift().bfill().fillna(0))['date'].transform(lambda x: (x - x.iloc[0]).dt.days)
    df['time_since_last_low'] = df.groupby(df['local_lows'].cumsum().shift().bfill().fillna(0))['date'].transform(lambda x: (x - x.iloc[0]).dt.days)

    df['price_reversal_indicator'] = (df['time_since_last_high'] - df['time_since_last_low']).clip(lower=0)

    # Combine Metrics
    df['factor'] = (df['short_term_volume_adjusted_return'] - df['long_term_volume_adjusted_return']) + df['price_reversal_indicator']
    factor = df['factor'].dropna()

    return factor
