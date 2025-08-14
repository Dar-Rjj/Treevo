import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Returns
    df['daily_returns'] = df['close'].pct_change()

    # Compute Weighted Average Return
    window_size = 10
    df['volume_moving_sum'] = df['volume'].rolling(window=window_size).sum()
    df['normalized_weights'] = df['volume'] / df['volume_moving_sum']
    df['weighted_returns'] = df['daily_returns'] * df['normalized_weights']

    # Smoothing Process
    smoothing_factor = 0.3
    df['smoothed_returns'] = df['weighted_returns'].ewm(alpha=smoothing_factor, adjust=False).mean()

    # Combined Factor Components
    span = 14

    # 14-Period Exponential Moving Averages
    df['high_ema'] = df['high'].ewm(span=span, adjust=False).mean()
    df['low_ema'] = df['low'].ewm(span=span, adjust=False).mean()
    df['close_ema'] = df['close'].ewm(span=span, adjust=False).mean()
    df['open_ema'] = df['open'].ewm(span=span, adjust=False).mean()

    # Compute 14-Period Price Envelopes
    df['max_price'] = df[['high', 'close']].max(axis=1)
    df['min_price'] = df[['low', 'close']].min(axis=1)
    df['max_price_ema'] = df['max_price'].ewm(span=span, adjust=False).mean()
    df['min_price_ema'] = df['min_price'].ewm(span=span, adjust=False).mean()
    df['envelope_distance'] = df['max_price_ema'] - df['min_price_ema']
    df['volume_smoothed'] = (df['envelope_distance'] * df['volume']).rolling(window=span).mean()

    # Construct Momentum Oscillator
    df['smoothed_positive_momentum'] = np.where(
        (df['high_ema'] - df['close_ema']) > 0,
        (df['high_ema'] - df['close_ema']) * df['volume_smoothed'],
        0
    )
    df['smoothed_negative_momentum'] = np.where(
        (df['low_ema'] - df['close_ema']) < 0,
        (df['low_ema'] - df['close_ema']) * df['volume_smoothed'],
        0
    )
    df['momentum_indicator'] = df['smoothed_positive_momentum'] - df['smoothed_negative_momentum']

    # Final Alpha Factor
    final_alpha_factor = df['smoothed_returns'] + df['momentum_indicator'] + 0.01
    threshold = 0.05
    final_alpha_factor = np.where(final_alpha_factor.abs() > threshold, final_alpha_factor, 0)

    return final_alpha_factor
