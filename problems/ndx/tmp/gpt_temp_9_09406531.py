import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute Daily Log Return
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate Volume-Weighted Exponential Moving Average (VWEMA) of Returns
    alpha_5 = 2 / (1 + 5)
    alpha_20 = 2 / (1 + 20)
    alpha_60 = 2 / (1 + 60)
    
    def vwema(series, volume, alpha):
        return series.ewm(alpha=alpha, adjust=False).mean() * (volume / volume.rolling(window=1).sum())
    
    df['vwema_5'] = vwema(df['log_return'], df['volume'], alpha_5)
    df['vwema_20'] = vwema(df['log_return'], df['volume'], alpha_20)
    df['vwema_60'] = vwema(df['log_return'], df['volume'], alpha_60)
    
    # Identify Volume Spikes
    df['vol_ma_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 1.5 * df['vol_ma_20'])
    
    # Adjust Daily Log Returns with Volume Spike
    df['adjusted_log_return'] = df['log_return'] * (2 if df['volume_spike'] else 1)
    
    # Calculate Short-Term and Long-Term Momentum Differentials
    df['momentum_diff_st'] = df['vwema_5'] - df['vwema_20']
    df['momentum_diff_lt'] = df['vwema_20'] - df['vwema_60']
    
    # Combine Momentum Differentials to form VWMAI
    df['vwmai'] = 0.4 * df['momentum_diff_st'] + 0.6 * df['momentum_diff_lt']
    
    # Compute 14-Period Exponential Moving Averages
    df['high_ema_14'] = df['high'].ewm(span=14, adjust=False).mean()
    df['low_ema_14'] = df['low'].ewm(span=14, adjust=False).mean()
    df['close_ema_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['open_ema_14'] = df['open'].ewm(span=14, adjust=False).mean()
    
    # Compute 14-Period Price Envelopes
    df['max_price'] = df[['high', 'close']].max(axis=1).ewm(span=14, adjust=False).mean()
    df['min_price'] = df[['low', 'close']].min(axis=1).ewm(span=14, adjust=False).mean()
    df['envelope_distance'] = df['max_price'] - df['min_price']
    df['normalized_envelope'] = df['envelope_distance'].apply(lambda x: x / df['envelope_distance'].iloc[-1])
    df['volume_smoothed'] = (df['normalized_envelope'] * df['volume']).rolling(window=14).mean()
    
    # Construct Momentum Oscillator
    df['smoothed_positive_momentum'] = ((df['high_ema_14'] - df['close_ema_14']) * df['volume_smoothed']).apply(lambda x: max(x, 0))
    df['smoothed_negative_momentum'] = ((df['low_ema_14'] - df['close_ema_14']) * df['volume_smoothed']).apply(lambda x: min(x, 0))
    df['momentum_indicator'] = df['smoothed_positive_momentum'] - df['smoothed_negative_momentum']
    
    # Final Alpha Factor
    df['final_alpha_factor'] = df['vwmai'] + df['momentum_indicator'] + 0.01
    
    return df['final_alpha_factor']
