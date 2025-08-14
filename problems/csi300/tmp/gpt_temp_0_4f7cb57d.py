import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import zscore

def heuristics_v2(df):
    # Calculate Price Momentum
    def price_momentum(df, n_days):
        return df['close'].pct_change(periods=n_days)

    def ema_price_momentum(df, n_days, m_days):
        pct_change = price_momentum(df, n_days)
        return pct_change.ewm(span=m_days, adjust=False).mean()
    
    # Calculate Volume Reversal
    def volume_change(df, lag):
        return df['volume'].pct_change(periods=lag)
    
    def cumulative_zscore_volume_change(df, lookback, lag):
        vol_change = volume_change(df, lag)
        return vol_change.rolling(window=lookback).apply(lambda x: zscore(x)[0], raw=False)
    
    # Combine Price Momentum and Volume Reversal
    def combine_factors(price_momentum, volume_reversal):
        combined = price_momentum * volume_reversal
        combined = combined.apply(lambda x: x if abs(x) >= 0.5 else 0)
        return combined
    
    # Compute N-day close price percent change
    price_momentum_5 = price_momentum(df, 5)
    price_momentum_10 = price_momentum(df, 10)
    price_momentum_21 = price_momentum(df, 21)
    
    # Compute M-day EMA of N-day close price percent change
    ema_price_momentum_5_5 = ema_price_momentum(df, 5, 5)
    ema_price_momentum_10_10 = ema_price_momentum(df, 10, 10)
    ema_price_momentum_21_21 = ema_price_momentum(df, 21, 21)
    
    # Average the EMA of price momentum
    avg_price_momentum = (ema_price_momentum_5_5 + ema_price_momentum_10_10 + ema_price_momentum_21_21) / 3
    
    # Compute volume change
    volume_change_1 = volume_change(df, 1)
    volume_change_2 = volume_change(df, 2)
    volume_change_5 = volume_change(df, 5)
    
    # Compute cumulative Z-score of volume change
    cum_zscore_vol_change_20_1 = cumulative_zscore_volume_change(df, 20, 1)
    cum_zscore_vol_change_20_2 = cumulative_zscore_volume_change(df, 20, 2)
    cum_zscore_vol_change_20_5 = cumulative_zscore_volume_change(df, 20, 5)
    
    # Average the cumulative Z-score of volume change
    avg_cum_zscore_vol_change = (cum_zscore_vol_change_20_1 + cum_zscore_vol_change_20_2 + cum_zscore_vol_change_20_5) / 3
    
    # Combine Price Momentum and Volume Reversal
    alpha_factor = combine_factors(avg_price_momentum, avg_cum_zscore_vol_change)
    
    return alpha_factor
