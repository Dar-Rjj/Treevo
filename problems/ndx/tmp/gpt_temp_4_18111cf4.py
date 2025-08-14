import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate Daily Momentum
    df['daily_momentum'] = df['close'].diff()

    # Identify Volume Spikes
    df['volume_20_day_avg'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['volume_20_day_avg'])

    # Adjust Daily Momentum by Volume Spike
    df['adjusted_daily_momentum'] = df['daily_momentum'] * (2.5 if df['volume_spike'] else 1)

    # Combine High-Low Range and Close-to-Open Return
    df['high_low_range'] = df['high'] - df['low']
    df['close_open_return'] = df['close'] - df['open']
    df['combined_factor'] = df['high_low_range'] + df['close_open_return']

    # Adjust by Volume
    df['combined_factor_adjusted'] = df['combined_factor'] / df['volume_20_day_avg']

    # Calculate 14-Period Exponential Moving Averages
    df['high_ema'] = df['high'].ewm(span=14, adjust=False).mean()
    df['low_ema'] = df['low'].ewm(span=14, adjust=False).mean()
    df['close_ema'] = df['close'].ewm(span=14, adjust=False).mean()
    df['open_ema'] = df['open'].ewm(span=14, adjust=False).mean()

    # Compute 14-Period Price Envelopes
    df['max_price'] = df[['high', 'open']].max(axis=1).ewm(span=14, adjust=False).mean()
    df['min_price'] = df[['low', 'open']].min(axis=1).ewm(span=14, adjust=False).mean()
    df['envelope_distance'] = df['max_price'] - df['min_price']
    df['volume_smoothed'] = (df['envelope_distance'] * df['volume']).rolling(window=14).mean()

    # Construct Volume-Weighted Momentum Oscillator
    df['positive_momentum'] = (df['high_ema'] - df['close_ema']) * df['volume_smoothed']
    df['negative_momentum'] = (df['low_ema'] - df['close_ema']) * df['volume_smoothed']
    df['smoothed_positive_momentum'] = df['positive_momentum'].apply(lambda x: x if x > 0 else 0).rolling(window=14).sum()
    df['smoothed_negative_momentum'] = df['negative_momentum'].apply(lambda x: x if x < 0 else 0).rolling(window=14).sum()
    df['momentum_indicator'] = df['smoothed_positive_momentum'] - abs(df['smoothed_negative_momentum'])

    # Final Alpha Factor
    df['final_alpha_factor'] = 0.7 * df['combined_factor_adjusted'] + 0.3 * df['momentum_indicator']
    df['final_alpha_factor'] = df['final_alpha_factor'].where(abs(df['final_alpha_factor']) > 0.01, 0)

    return df['final_alpha_factor']
