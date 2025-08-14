import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate 14-Period Exponential Moving Averages
    ema_high = df['high'].ewm(span=14, adjust=False).mean()
    ema_low = df['low'].ewm(span=14, adjust=False).mean()
    ema_close = df['close'].ewm(span=14, adjust=False).mean()
    ema_open = df['open'].ewm(span=14, adjust=False).mean()
    
    # Compute Daily Log Return
    log_return = np.log(df['close'] / df['close'].shift(1))
    
    # Calculate Volume-Weighted Exponential Moving Average (VWEMA) of Returns
    def vwema(series, volume, alpha):
        return series.ewm(alpha=alpha, adjust=False).mean().mul(volume).ewm(alpha=alpha, adjust=False).mean()
    
    vwema_7 = vwema(log_return, df['volume'], alpha=2/(1+7))
    vwema_20 = vwema(log_return, df['volume'], alpha=2/(1+20))
    vwema_60 = vwema(log_return, df['volume'], alpha=2/(1+60))
    
    # Calculate Short-Term and Long-Term Momentum Differentials
    short_term_diff = vwema_7 - vwema_20
    long_term_diff = vwema_20 - vwema_60
    
    # Combine Momentum Differentials to form VWMAI
    vwmai = 0.5 * short_term_diff + 0.5 * long_term_diff
    
    # Compute 14-Period Price Envelopes
    max_price = df[['high', 'close']].max(axis=1).ewm(span=14, adjust=False).mean()
    min_price = df[['low', 'close']].min(axis=1).ewm(span=14, adjust=False).mean()
    envelope_distance = max_price - min_price
    volume_smoothing_factor = (max_price - min_price) * df['volume']
    volume_smoothing_factor = volume_smoothing_factor.rolling(window=14).mean()
    
    # Construct Volume-Enhanced and -Weighted Momentum Oscillator
    smoothed_positive_momentum = (ema_high - ema_close) * volume_smoothing_factor
    smoothed_positive_momentum = np.where(smoothed_positive_momentum > 0, smoothed_positive_momentum, 0)
    
    smoothed_negative_momentum = (ema_low - ema_close) * volume_smoothing_factor
    smoothed_negative_momentum = np.where(smoothed_negative_momentum < 0, smoothed_negative_momentum, 0)
    
    net_momentum = smoothed_positive_momentum - smoothed_negative_momentum
    
    # Identify Volume Spikes
    volume_30d_ma = df['volume'].rolling(window=30).mean()
    volume_spike = df['volume'] > 2.0 * volume_30d_ma
    
    # Final Factor Value
    factor_value = net_momentum * (3.0 if volume_spike else 1.0) + vwmai
    
    # Combine VWMAI and VEMAO
    final_factor = 0.7 * vwmai + 0.3 * factor_value
    
    return final_factor
